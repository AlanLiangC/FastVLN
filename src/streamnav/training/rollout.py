import json
import time
from contextlib import nullcontext
from dataclasses import replace
from typing import Any, cast

import torch

from streamnav.contracts.action import NavigationAction
from streamnav.errors import OracleUnavailableError
from streamnav.models.qwen35_kda.cache import clone_state
from streamnav.training.auxiliary_il import AuxiliaryILController
from streamnav.training.dagger import behavior_log_prob, select_env_action
from streamnav.training.rollout_buffer import RecurrentRolloutBuffer


class RolloutCollector:
    def __init__(self, policy, envs, sources, config, auxiliary_controller=None):
        self.policy, self.envs, self.sources, self.config = policy, envs, sources, config
        self.observations = None
        self.states = None
        self.update_index = 0
        self.warmup_steps = 0
        self.curriculum_fallbacks = 0
        self.reset_elapsed = 0.0
        self.cache_vision = config.get("cache_frozen_vision", False)
        self.last_visual_embeddings = None
        self.perception_enabled = getattr(policy, "perception", None) is not None
        self.auxiliary = auxiliary_controller or AuxiliaryILController(
            config.get("auxiliary_il"), len(sources)
        )
        if self.cache_vision and any(p.requires_grad for p in policy.backbone.vision.parameters()):
            raise ValueError("Frozen-vision caching requires a fully frozen visual encoder")

    def record_reset(self, observation):
        self.warmup_steps += observation.get("curriculum_warmup_steps", 0)
        self.curriculum_fallbacks += int("curriculum_fallback" in observation)

    def sample_episode(self, index):
        episode = self.sources[index].sample_episode()
        curriculum = self.config.get("curriculum", {})
        duration = curriculum.get("duration_updates", 0)
        if curriculum.get("enabled", False) and self.update_index < duration:
            if torch.rand(()).item() < curriculum["probability"]:
                fraction = self.update_index / duration
                distance = curriculum["start_distance"] + fraction * (
                    curriculum["end_distance"] - curriculum["start_distance"]
                )
                episode = replace(
                    episode, metadata={**episode.metadata, "training_warm_start_distance": distance}
                )
        return episode

    def forward_observations(self, rgb, states):
        # BF16 vision/GEMM kernels may differ with batch size. Match the replay
        # batch width without changing four environments or their trajectories.
        size = self.config.get("rollout_inference_batch_size", len(states))
        outputs, visual = [], []
        for start in range(0, len(states), size):
            frames, batch_states = rgb[start : start + size], states[start : start + size]
            if self.cache_vision:
                embeddings = self.policy.backbone.encode_vision(frames)
                outputs.append(
                    self.policy.forward_batch(frames, batch_states, visual_embeddings=embeddings)
                )
                visual.append(embeddings)
            else:
                outputs.append(self.policy.forward_batch(frames, batch_states))
        self.last_visual_embeddings = torch.cat(visual) if visual else None
        return (
            torch.cat([output[0] for output in outputs]),
            torch.cat([output[1] for output in outputs]),
            [state for output in outputs for state in output[2]],
        )

    @torch.no_grad()
    def reset(self):
        episodes = [self.sample_episode(i) for i in range(len(self.sources))]
        start = time.monotonic()
        self.observations = self.envs.reset(episodes)
        self.reset_elapsed += time.monotonic() - start
        for observation in self.observations:
            self.record_reset(observation)
        self.states = [self.policy.start_episode(e.uid, e.goal_text) for e in episodes]
        for i in range(self.auxiliary.num_envs):
            self.auxiliary.reset(i)

    @torch.no_grad()
    def collect(self, beta):
        start = time.monotonic()
        self.policy.eval()
        self.warmup_steps = 0
        self.curriculum_fallbacks = 0
        self.reset_elapsed = 0.0
        if self.states is None:
            self.reset()
        else:
            # A reset on the last rollout step prefills under the old weights.
            # Optimization happens before its first RGB frame. Refresh this
            # unused prefix so collection matches differentiable replay, which
            # recomputes episode starts with the current weights. Preserve all
            # ongoing episode memories across update boundaries.
            self.states = [
                self.policy.start_episode(s.episode_id, s.instruction) if s.step_index == 0 else s
                for s in self.states
            ]
        assert self.states is not None and self.observations is not None
        cfg = self.config
        buffer = RecurrentRolloutBuffer(
            cfg["rollout_steps"],
            len(self.states),
            self.observations[0]["rgb"].shape,
            cfg["sequence_length"],
            beta,
        )
        if self.perception_enabled:
            buffer.enable_perception()
        if self.auxiliary.num_envs:
            buffer.ppo_mask[:, : self.auxiliary.num_envs] = False
            buffer.replay_log_probs = torch.empty_like(buffer.old_log_probs)
        for t in range(buffer.steps):
            supervision = self.envs.get_oracle_supervisions() if self.perception_enabled else None
            experts = (
                [
                    entry
                    if isinstance(entry, OracleUnavailableError)
                    else NavigationAction(entry["action"])
                    for entry in supervision
                ]
                if supervision is not None
                else self.envs.get_oracle_actions()
            )
            for i, expert in enumerate(experts):
                if not isinstance(expert, OracleUnavailableError):
                    continue
                # Never invent an expert label. End the collected prefix as a
                # truncation, bootstrap from its real final frame, and log it.
                if t > 0 and not bool(buffer.dones[t - 1, i]):
                    value = self.policy.forward_step(
                        self.observations[i]["rgb"], self.states[i]
                    ).value
                    buffer.dones[t - 1, i] = True
                    buffer.timeout_bootstrap[t - 1, i] = value.cpu()
                for attempt in range(3):
                    failure = {"episode_id": self.states[i].episode_id, "reason": str(expert)}
                    buffer.oracle_failures.append(failure)
                    print(json.dumps({"oracle_episode_skipped": failure}), flush=True)
                    episode = self.sample_episode(i)
                    reset_start = time.monotonic()
                    self.observations[i] = self.envs.clients[i].reset(episode)
                    self.reset_elapsed += time.monotonic() - reset_start
                    self.record_reset(self.observations[i])
                    self.states[i] = self.policy.start_episode(episode.uid, episode.goal_text)
                    self.auxiliary.reset(i)
                    buffer.resets[(i, t)] = (episode.uid, episode.goal_text)
                    try:
                        if supervision is not None:
                            supervision[i] = self.envs.clients[i].get_oracle_supervision()
                            experts[i] = NavigationAction(supervision[i]["action"])
                        else:
                            experts[i] = self.envs.clients[i].get_oracle_action()
                        break
                    except OracleUnavailableError as exc:
                        expert = exc
                else:
                    raise OracleUnavailableError(
                        "Three consecutive episodes failed oracle recovery"
                    )
            if supervision is not None:
                for i, entry in enumerate(supervision):
                    labels = entry["perception"]
                    observation = self.observations[i]
                    if (
                        labels["episode_id"] != self.states[i].episode_id
                        or labels["frame_id"] != observation["frame_id"]
                    ):
                        raise ValueError("Perception labels do not match the pre-action RGB frame")
                    buffer.store_perception(t, i, labels)
            buffer.save_boundary(t, self.states)
            rgb = torch.stack([o["rgb"] for o in self.observations])
            logits, values, states = self.forward_observations(rgb, self.states)
            if self.last_visual_embeddings is not None:
                # Store only frozen visual outputs. Text/markers/NAV are rebuilt
                # under the current trainable parameters during every replay.
                if buffer.visual_embeddings is None:
                    buffer.visual_embeddings = torch.empty(
                        (buffer.steps, *self.last_visual_embeddings.shape),
                        dtype=self.last_visual_embeddings.dtype,
                    )
                buffer.visual_embeddings[t].copy_(self.last_visual_embeddings.cpu())
            dist = self.policy.distribution.build(logits)
            actions = dist.sample()
            expert = torch.tensor(experts, device=logits.device)
            executed, used_expert = select_env_action(actions, expert, beta)
            greedy = logits.argmax(-1)
            if self.auxiliary.num_envs:
                # beta=0 returns the sampled tensor itself. Preserve that
                # sample when a deterministic auxiliary action replaces it.
                executed = executed.clone()
                greedy_actions = greedy.tolist()
                for i in range(self.auxiliary.num_envs):
                    action, followed = self.auxiliary.select(
                        i, greedy_actions[i], int(cast(NavigationAction, experts[i]))
                    )
                    executed[i], used_expert[i] = action, followed
                    buffer.auxiliary_expert_steps += int(followed)
                    buffer.auxiliary_recovery_triggers += int(
                        self.auxiliary.observe(i, action, followed)
                    )
                # Log the actual deterministic auxiliary behavior (probability
                # one). Its model probabilities are stored separately for the
                # numerical replay check; auxiliary frames are excluded from PPO.
                assert buffer.replay_log_probs is not None
                buffer.replay_log_probs[t].copy_(dist.log_prob(executed).cpu())
            old_log_probs = behavior_log_prob(logits, executed, expert, beta)
            if self.auxiliary.num_envs:
                old_log_probs[: self.auxiliary.num_envs] = 0
            executed_actions = executed.tolist()
            results = self.envs.step([NavigationAction(a) for a in executed_actions])
            for source in self.sources:
                if hasattr(source, "step_taken"):
                    source.step_taken()
            buffer.observations[t].copy_(rgb)
            for name, value in (
                ("actions", actions),
                ("greedy_actions", greedy),
                ("executed_actions", executed),
                ("expert_actions", expert),
                ("used_expert", used_expert),
                ("old_values", values),
                ("entropies", dist.entropy()),
                ("stop_probabilities", dist.probs[:, int(NavigationAction.STOP)]),
                ("old_policy_log_probs", dist.log_prob(actions)),
                ("old_log_probs", old_log_probs),
            ):
                getattr(buffer, name)[t].copy_(value.cpu())
            reset_episodes = {}
            for i, result in enumerate(results):
                buffer.oracle_navigation_repairs += max(
                    0,
                    result.get("oracle_navigation_repairs", 0)
                    - self.observations[i].get("oracle_navigation_repairs", 0),
                )
                if cfg.get("filter_blocked_forward_labels", False):
                    # Supervision validity only: the sampled transition, PPO
                    # probability/reward and recurrent history stay intact.
                    if (
                        experts[i] == NavigationAction.MOVE_FORWARD
                        and executed_actions[i] == NavigationAction.MOVE_FORWARD
                        and result["collision"]
                        and result["displacement"]
                        <= self.envs.config.get("forward_step", 0.25) * 0.05
                    ):
                        buffer.il_mask[t, i] = False
                        if buffer.perception_targets is not None:
                            buffer.perception_targets["apos"]["valid"][t, i] = False
                buffer.rewards[t, i] = result["reward"]
                buffer.dones[t, i] = result["done"]
                buffer.collisions += int(result["collision"])
                if i < self.auxiliary.num_envs:
                    buffer.auxiliary_collisions += int(result["collision"])
                if result["truncated"] and cfg.get("ppo", {}).get("bootstrap_time_limits", False):
                    # Bootstrap from the final frame of this episode, before reset.
                    buffer.timeout_bootstrap[t, i] = self.policy.forward_step(
                        result["rgb"], states[i]
                    ).value.cpu()
                if result["done"]:
                    episode_metrics = (
                        buffer.auxiliary_episode_metrics
                        if i < self.auxiliary.num_envs
                        else buffer.episode_metrics
                    )
                    episode_metrics.append(result["metrics"])
                    episode = self.sample_episode(i)
                    self.auxiliary.reset(i)
                    reset_episodes[i] = episode
                    states[i] = self.policy.start_episode(episode.uid, episode.goal_text)
                    if t + 1 < buffer.steps:
                        buffer.resets[(i, t + 1)] = (episode.uid, episode.goal_text)
            if reset_episodes:
                reset_start = time.monotonic()
                for i, observation in self.envs.reset_at(reset_episodes).items():
                    results[i] = observation
                    self.record_reset(observation)
                self.reset_elapsed += time.monotonic() - reset_start
            self.states, self.observations = states, results
        _, final_values, _ = self.forward_observations(
            torch.stack([o["rgb"] for o in self.observations]), self.states
        )
        buffer.last_values = final_values.cpu()
        buffer.elapsed_s = time.monotonic() - start
        buffer.curriculum_warmup_steps = self.warmup_steps
        buffer.curriculum_fallbacks = self.curriculum_fallbacks
        buffer.reset_seconds = self.reset_elapsed
        # No graph and no cache alias survives into optimization.
        self.states = [clone_state(s) for s in states]
        return buffer


def replay_sequences(policy, buffer, sequences, *, cache_embeddings=False, pack_frames=1):
    if pack_frames < 1:
        raise ValueError("Replay frame packing must be positive")
    context = policy.backbone.reuse_token_embeddings() if cache_embeddings else nullcontext()
    with context:
        if pack_frames > 1:
            return _replay_packed_chat(policy, buffer, sequences, pack_frames)
        return _replay_sequences(policy, buffer, sequences)


def _replay_packed_chat(policy, buffer, sequences, pack_frames):
    """Concatenate causal chat frames, splitting at every episode reset.

    Packing changes BF16 chunk boundaries, so it is opt-in and must pass the
    same pre-update behavior-probability check as frame-by-frame replay.
    Initial memories remain detached; gradients flow through all packed blocks.
    """
    body = policy.backbone
    if body.goal_conditioning != "chat_query" or buffer.visual_embeddings is None:
        raise ValueError("Packed replay requires chat_query and frozen visual embeddings")
    if any(p.requires_grad for p in body.vision.parameters()):
        raise ValueError("Cannot replay cached embeddings after unfreezing vision")
    length = sequences[0].stop - sequences[0].start
    if any(s.stop - s.start != length for s in sequences):
        raise ValueError("Replay sequences must have equal frame counts")
    readouts = []
    for seq in sequences:
        state = clone_state(buffer.initial_states[(seq.env, seq.start)])
        visual = buffer.visual_embeddings[seq.start : seq.stop, seq.env].to(body.device)
        frames, offset = [], 0
        while offset < length:
            step = seq.start + offset
            reset = buffer.resets.get((seq.env, step))
            if offset == 0 and state.step_index == 0:
                reset = (state.episode_id, state.instruction)
            if reset is not None:
                state = policy.start_episode(*reset)
            stop = min(offset + pack_frames, length)
            stop = next(
                (i for i in range(offset + 1, stop) if (seq.env, seq.start + i) in buffer.resets),
                stop,
            )
            tokens = body.encode_chat_visual_tokens(
                visual[offset:stop],
                [state.instruction] * (stop - offset),
                [state.step_index == 0] + [False] * (stop - offset - 1),
            )
            hidden, cache = body.recurrent_forward(torch.cat(tokens).unsqueeze(0), state.kda_cache)
            ends = torch.tensor([t.shape[0] for t in tokens], device=body.device).cumsum(0) - 1
            frames.append(hidden[0, ends])
            state = replace(state, kda_cache=cache, step_index=state.step_index + stop - offset)
            offset = stop
        readouts.append(torch.cat(frames))
    hidden = torch.stack(readouts, dim=1)
    # Retain collection's head batch width; only the recurrent body is packed.
    outputs = [policy.readout(frame) for frame in hidden]
    result = (
        torch.stack([x[0] for x in outputs]).float(),
        torch.stack([x[1] for x in outputs]).float(),
    )
    if policy.perception is not None:
        predictions = {
            name: torch.stack([x[2][name] for x in outputs]) for name in ("apos", "opos", "arrival")
        }
        return (*result, predictions)
    return result


def _replay_sequences(policy, buffer, sequences):
    if buffer.visual_embeddings is not None and any(
        p.requires_grad for p in policy.backbone.vision.parameters()
    ):
        raise ValueError("Cannot replay cached embeddings after unfreezing vision")
    states = [clone_state(buffer.initial_states[(s.env, s.start)]) for s in sequences]
    logits, values, predictions = [], [], []
    perception_enabled = getattr(policy, "perception", None) is not None
    for offset in range(sequences[0].stop - sequences[0].start):
        frames = []
        for i, seq in enumerate(sequences):
            step = seq.start + offset
            reset = buffer.resets.get((seq.env, step))
            if offset == 0 and states[i].step_index == 0:
                reset = (states[i].episode_id, states[i].instruction)
            if reset is not None:
                states[i] = policy.start_episode(*reset)
            frames.append(buffer.observations[step, seq.env])
        kwargs: dict[str, Any] = {"return_perception": True} if perception_enabled else {}
        if buffer.visual_embeddings is not None:
            kwargs["visual_embeddings"] = torch.stack(
                [buffer.visual_embeddings[s.start + offset, s.env] for s in sequences]
            ).to(policy.backbone.device)
        output = policy.forward_batch(torch.stack(frames), states, **kwargs)
        step_logits, step_values, states = output[:3]
        if perception_enabled:
            predictions.append(output[3])
        logits.append(step_logits)
        values.append(step_values)
    result = (torch.stack(logits), torch.stack(values))
    if perception_enabled:
        return (
            *result,
            {
                name: torch.stack([p[name] for p in predictions])
                for name in ("apos", "opos", "arrival")
            },
        )
    return result


class SequenceReplay(torch.nn.Module):
    """One DDP forward encompasses the whole differentiable recurrent sequence."""

    def __init__(self, policy, cache_embeddings=False, pack_frames=1):
        super().__init__()
        self.policy = policy
        self.cache_embeddings = cache_embeddings
        self.pack_frames = pack_frames

    def forward(self, buffer, sequences):
        return replay_sequences(
            self.policy,
            buffer,
            sequences,
            cache_embeddings=self.cache_embeddings,
            pack_frames=self.pack_frames,
        )

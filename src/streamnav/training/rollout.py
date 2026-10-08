import json
import time
from dataclasses import replace

import torch

from streamnav.contracts.action import NavigationAction
from streamnav.errors import OracleUnavailableError
from streamnav.models.qwen35_kda.cache import clone_state
from streamnav.training.dagger import behavior_log_prob, select_env_action
from streamnav.training.rollout_buffer import RecurrentRolloutBuffer


class RolloutCollector:
    def __init__(self, policy, envs, sources, config):
        self.policy, self.envs, self.sources, self.config = policy, envs, sources, config
        self.observations = None
        self.states = None
        self.update_index = 0
        self.warmup_steps = 0
        self.curriculum_fallbacks = 0
        self.reset_elapsed = 0.0
        self.cache_vision = config.get("cache_frozen_vision", False)
        self.last_visual_embeddings = None
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
        for t in range(buffer.steps):
            experts = self.envs.get_oracle_actions()
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
                    buffer.resets[(i, t)] = (episode.uid, episode.goal_text)
                    try:
                        experts[i] = self.envs.clients[i].get_oracle_action()
                        break
                    except OracleUnavailableError as exc:
                        expert = exc
                else:
                    raise OracleUnavailableError(
                        "Three consecutive episodes failed oracle recovery"
                    )
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
            results = self.envs.step([NavigationAction(a) for a in executed.tolist()])
            for source in self.sources:
                if hasattr(source, "step_taken"):
                    source.step_taken()
            buffer.observations[t].copy_(rgb)
            for name, value in (
                ("actions", actions),
                ("greedy_actions", logits.argmax(-1)),
                ("executed_actions", executed),
                ("expert_actions", expert),
                ("used_expert", used_expert),
                ("old_values", values),
                ("entropies", dist.entropy()),
                ("stop_probabilities", dist.probs[:, int(NavigationAction.STOP)]),
                ("old_policy_log_probs", dist.log_prob(actions)),
                ("old_log_probs", behavior_log_prob(logits, executed, expert, beta)),
            ):
                getattr(buffer, name)[t].copy_(value.cpu())
            reset_episodes = {}
            for i, result in enumerate(results):
                buffer.rewards[t, i] = result["reward"]
                buffer.dones[t, i] = result["done"]
                buffer.collisions += int(result["collision"])
                if result["truncated"] and cfg.get("ppo", {}).get("bootstrap_time_limits", False):
                    # Bootstrap from the final frame of this episode, before reset.
                    buffer.timeout_bootstrap[t, i] = self.policy.forward_step(
                        result["rgb"], states[i]
                    ).value.cpu()
                if result["done"]:
                    buffer.episode_metrics.append(result["metrics"])
                    episode = self.sample_episode(i)
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


def replay_sequences(policy, buffer, sequences):
    if buffer.visual_embeddings is not None and any(
        p.requires_grad for p in policy.backbone.vision.parameters()
    ):
        raise ValueError("Cannot replay cached embeddings after unfreezing vision")
    states = [clone_state(buffer.initial_states[(s.env, s.start)]) for s in sequences]
    logits, values = [], []
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
        if buffer.visual_embeddings is None:
            step_logits, step_values, states = policy.forward_batch(torch.stack(frames), states)
        else:
            visual = torch.stack(
                [buffer.visual_embeddings[s.start + offset, s.env] for s in sequences]
            ).to(policy.backbone.device)
            step_logits, step_values, states = policy.forward_batch(
                torch.stack(frames), states, visual_embeddings=visual
            )
        logits.append(step_logits)
        values.append(step_values)
    return torch.stack(logits), torch.stack(values)


class SequenceReplay(torch.nn.Module):
    """One DDP forward encompasses the whole differentiable recurrent sequence."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, buffer, sequences):
        return replay_sequences(self.policy, buffer, sequences)

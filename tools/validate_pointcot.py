"""GPU acceptance: native equivalence, recurrent PPO replay, resets and learning."""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from streamnav.models.point_cot.policy import Qwen35PointCoTPolicy
from streamnav.training.grounding_trainer import optimizer_for
from streamnav.training.point_supervision import supervised_fragment
from streamnav.utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pointcot.yaml")
    parser.add_argument("--output", default="runtime/reports/pointcot_validation.json")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    seed_everything(config["seed"])
    policy = Qwen35PointCoTPolicy(config).train()
    frame = torch.randint(0, 255, (270, 480, 3), dtype=torch.uint8)
    features = [
        policy.frame_features(frame, goal)[0]
        for goal in ("Find a chair.", "Find a small red lamp.")
    ]
    initial = policy.start_episode("one", "Find a chair.")
    with torch.no_grad():
        context = policy.begin(features[0], initial, native_only=True)
        text = policy.native.model.language_model
        x = text.embed_tokens(features[0].ids)
        x = x.masked_scatter(
            (features[0].ids == policy.native.config.image_token_id).unsqueeze(-1),
            features[0].visual.float(),
        )
        expected = text(
            inputs_embeds=x, position_ids=features[0].positions, use_cache=False
        ).last_hidden_state[:, -1]
        native_error = float((context.hidden - expected).abs().max())
        assert native_error < 0.001, native_error
        # Nonzero memory makes cache equivalence an actual recurrent test.
        for memory in policy.memories.values():
            memory.residual_scale.fill_(0.05)
        a = policy.step(features[0], initial)
        b = policy.step(features[1], a.state)
        c = policy.step(features[0], policy.start_episode("two", "Find a chair."))
    tokens = [(s.apos, s.opos, s.action) for s in (a, b, c)]
    replay = policy.replay_fragment(
        [*features, features[0]], initial, tokens, resets={2: ("two", "Find a chair.")}
    )
    old = torch.stack([s.log_prob for s in (a, b, c)])
    error = float((replay.log_prob.detach() - old).abs().max())
    assert error < config["trainer"]["replay_tolerance"], error
    loss = -replay.log_prob.mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in policy._all_trainable if p.grad is not None)
    assert any(
        p.grad is not None and bool(p.grad.abs().sum()) for p in policy.memories.parameters()
    )
    cache_bytes = sum(
        s.recurrent.numel() * s.recurrent.element_size()
        for s in c.state.kda_cache
        if s.recurrent is not None
    )
    # A small controlled fit verifies the new tied point rows and teacher prefixes.
    optimizer = optimizer_for(policy, config["grounding"]["learning_rates"])
    labels = [{"apos_targets": [900], "opos_targets": [400]}]
    history = []
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        step, apos, opos, action = supervised_fragment(
            policy,
            features[:1],
            initial,
            tokens[:1],
            labels,
            experts=torch.tensor([0], device=policy.device),
        )
        objective = apos.mean() + opos.mean() + action.mean()
        history.append(float(objective.detach()))
        objective.backward()
        torch.nn.utils.clip_grad_norm_(policy._all_trainable, 1, error_if_nonfinite=True)
        optimizer.step()
    assert history[-1] < history[0] * 0.8, history
    policy.eval()
    with torch.no_grad():
        for _ in range(3):
            policy.step(features[0], initial, deterministic=True)
        torch.cuda.synchronize()
        started = time.monotonic()
        for _ in range(10):
            policy.step(features[0], initial, deterministic=True)
        torch.cuda.synchronize()
        seconds = (time.monotonic() - started) / 10
    result = {
        "parameters": policy.parameter_summary(),
        "native_hidden_error_max": native_error,
        "replay_log_prob_error_max": error,
        "cache_bytes": cache_bytes,
        "controlled_loss": history,
        "decision_seconds_cached_vision": seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "passed": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()

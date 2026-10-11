"""Reproduce one evaluation shard with its original companions and batch widths."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from streamnav.contracts.action import NavigationAction
from streamnav.data.schema import HabitatEpisodeSource
from streamnav.envs.habitat_client import VectorHabitatEnvs
from streamnav.training.checkpoint import load_policy
from streamnav.utils.seed import seed_everything


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--disable-fusion", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    config.update(device=f"cuda:{args.gpu}", checkpoint=args.checkpoint)
    seed_everything(config["seed"])
    policy = load_policy(config, preserve_master_weights=True).eval()
    policy.batch_chat_body = config["trainer"]["batch_chat_body"]
    if args.disable_fusion:
        policy.perception.fusion_scale = 0.0
    source = HabitatEpisodeSource(config["eval"]["manifests"][2], seed=config["seed"])
    batch = [
        (i, e)
        for i, e in enumerate(source.evaluation_episodes(48, True, config["seed"]))
        if i % 8 == args.rank
    ]
    assert len(batch) == 6
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    envs = VectorHabitatEnvs(
        {**config["habitat"], "gpu_device_id": args.gpu, "perception_labels": False}, 6
    )
    traces = [[] for _ in batch]
    results = []
    try:
        observations = envs.reset([e for _, e in batch])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            states = [policy.start_episode(e.uid, e.goal_text) for _, e in batch]
        active = list(range(6))
        while active:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _, updated = policy.forward_batch(
                    torch.stack([observations[i]["rgb"] for i in active]).to(
                        policy.backbone.device
                    ),
                    [states[i] for i in active],
                )
            actions = [NavigationAction(x) for x in logits.argmax(-1).tolist()]
            probabilities = logits.softmax(-1).tolist()
            following = list(
                envs.executor.map(
                    lambda pair: envs.clients[pair[0]].step(pair[1]),
                    zip(active, actions, strict=True),
                )
            )
            remaining = []
            for i, action, state, after, probs in zip(
                active, actions, updated, following, probabilities, strict=True
            ):
                before = observations[i]
                states[i] = state
                observations[i] = after
                traces[i].append(
                    {
                        "frame_id": before["frame_id"],
                        "action": action.name,
                        "probabilities": probs,
                        "distance": before.get("geodesic_distance", before.get("distance")),
                    }
                )
                if after["done"]:
                    result = {
                        "episode_index": batch[i][0],
                        "goal": batch[i][1].goal_text,
                        **after["metrics"],
                    }
                    results.append(result)
                    print(json.dumps(result), flush=True)
                else:
                    remaining.append(i)
            active = remaining
    finally:
        envs.close()
    for (i, _), trace in zip(batch, traces, strict=True):
        (output / f"episode_{i:04d}.json").write_text(json.dumps(trace))
    (output / "results.json").write_text(
        json.dumps(
            {
                "scope": f"Only rank{args.rank}'s six unseen episodes, original six-environment batch; not full benchmark",
                "fusion_enabled": not args.disable_fusion,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()

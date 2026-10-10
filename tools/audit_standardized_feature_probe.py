"""Fit a six-action linear probe with training-only feature statistics.

Centering/scaling changes optimization and regularization, not linear capacity.
No backbone/NAV updates, semantic inputs or test-statistic fitting.
"""

import argparse
import json
from pathlib import Path

import torch
from audit_architecture_capacity import write_json
from audit_cross_scene_capacity import dataset, score
from torch import nn
from torch.nn import functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(args.data_root)
    clips = dataset(root)
    labels = torch.cat([c["actions"] for c in clips["train"]])
    records = {}
    for variant in args.variants:
        features = torch.load(
            root / f"fit_{variant}.features.pt", map_location="cpu", weights_only=False
        )["features"]
        train = features["train"]
        mean, scale = train.mean(0), train.std(0).clamp_min(1e-4)
        normalized = {k: (v - mean) / scale for k, v in features.items()}
        torch.manual_seed(10913)
        probe = nn.Linear(train.shape[-1], 6)
        nn.init.zeros_(probe.weight)
        nn.init.zeros_(probe.bias)
        optimizer = torch.optim.LBFGS(
            probe.parameters(), max_iter=200, tolerance_grad=1e-7, line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(probe(normalized["train"]), labels)
            loss = loss + 1e-3 * probe.weight.square().sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            records[variant] = {
                "scores": {k: score(probe(x), clips[k]) for k, x in normalized.items()},
                "training_feature_centered_rms_ratio": (
                    (train - mean).square().mean().sqrt() / train.square().mean().sqrt()
                ).item(),
                "train_goal_pair_relative_l2_mean": (
                    (train[::2] - train[1::2]).norm(dim=-1)
                    / train[::2].norm(dim=-1).clamp_min(1e-6)
                )
                .mean()
                .item(),
                "optimizer_iterations": optimizer.state[probe.weight]["n_iter"],
                "final_objective": (
                    F.cross_entropy(probe(normalized["train"]), labels)
                    + 1e-3 * probe.weight.square().sum()
                ).item(),
            }
        print(
            json.dumps(
                {
                    variant: {
                        k: v["stop_binary_accuracy"] for k, v in records[variant]["scores"].items()
                    }
                }
            ),
            flush=True,
        )
    write_json(
        args.output,
        {
            "protocol": __doc__,
            "regularization_weight_squared_l2": 1e-3,
            "records": records,
            "test_data_used_in_optimizer": False,
        },
    )


if __name__ == "__main__":
    main()

"""Fit matched frozen-feature heads on one or multiple stationary contexts.

Train-only feature statistics and labels; no backbone or navigation updates.
Same scenes are reused from prior diagnostics, so this is an adaptation control.
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
    parser.add_argument("--context-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    clips = dataset(Path(args.data_root))
    root = Path(args.context_root)
    records, probes = {}, {}
    for variant, filename in (
        ("native", "native_context.features.pt"),
        ("kda", "readout_context_with_train.features.pt"),
    ):
        features = torch.load(root / filename, map_location="cpu", weights_only=False)
        for name, contexts in (("single_frame", [1]), ("mixed_contexts", [1, 4, 16])):
            torch.manual_seed(10913)
            train = torch.cat([features["train"][i] for i in contexts])
            targets = torch.cat([c["actions"] for c in clips["train"]]).repeat(len(contexts))
            mean, scale = train.mean(0), train.std(0).clamp_min(1e-4)
            model = nn.Linear(train.shape[-1], 6)
            nn.init.zeros_(model.weight)
            nn.init.zeros_(model.bias)
            optimizer = torch.optim.LBFGS(
                model.parameters(), max_iter=200, tolerance_grad=1e-7, line_search_fn="strong_wolfe"
            )

            def closure():
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model((train - mean) / scale), targets)
                loss = loss + 1e-3 * model.weight.square().sum()
                loss.backward()
                return loss

            optimizer.step(closure)
            with torch.no_grad():
                scores = {
                    split: {
                        frame: score(model((x - mean) / scale), clips[split])
                        for frame, x in rows.items()
                    }
                    for split, rows in features.items()
                }
            key = variant + "/" + name
            records[key] = {
                "training_contexts": contexts,
                "training_feature_rows": train.shape[0],
                "optimizer_iterations": optimizer.state[model.weight]["n_iter"],
                "scores": scores,
            }
            probes[key] = {"mean": mean, "scale": scale, "head": model.state_dict()}
            print(
                json.dumps(
                    {
                        key: {
                            s: {i: r["stop_binary_accuracy"] for i, r in v.items()}
                            for s, v in scores.items()
                        }
                    }
                ),
                flush=True,
            )
    torch.save(probes, Path(args.output).with_suffix(".probes.pt"))
    write_json(args.output, {"protocol": __doc__, "records": records})


if __name__ == "__main__":
    main()

"""Calibrate six converted output projections using native attention outputs.

No action labels, semantic inputs, heldout frames or navigation optimizer state.
Fit one scalar per converted O matrix on native layer inputs: equal weight for
mean-token and final-NAV output error. This is initialization calibration, not
end-to-end distillation or an autonomous navigation improvement.
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
import yaml
from audit_architecture_capacity import configuration, write_json
from audit_native_goal_pair_fit import FrameReadout
from safetensors.torch import save_file

from streamnav.contracts.state import LayerState
from streamnav.utils.seed import seed_everything


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    seed_everything(10913)
    torch.cuda.set_device(args.device)
    clips = torch.load(Path(args.data_root) / "train.pt", map_location="cpu", weights_only=False)[
        "clips"
    ]
    teacher = FrameReadout(
        argparse.Namespace(checkpoint=args.checkpoint, device=args.device, variant="native")
    ).eval()
    student = FrameReadout(
        argparse.Namespace(checkpoint=args.checkpoint, device=args.device, variant="kda")
    ).eval()
    indices = [3, 7, 11, 15, 19, 23]
    moments = {i: torch.zeros(3, dtype=torch.float64, device=args.device) for i in indices}
    counts = dict.fromkeys(indices, 0)
    hooks = []
    for i in indices:
        native = teacher.body.language_model.layers[i].self_attn

        def measure(module, inputs, kwargs, output, index=i):
            x = kwargs.get("hidden_states", inputs[0] if inputs else None)
            reference = output[0].float()
            prediction, _ = student.body.layers[index].mixer(
                x, LayerState(None, None), mode="chunk"
            )
            prediction = prediction.float()
            # Equal example weights, no over-weighting longer goal prefixes.
            for s, t in ((prediction, reference), (prediction[:, -1:], reference[:, -1:])):
                moments[index] += torch.stack(
                    (
                        (s.double() * t).mean((1, 2)).sum(),
                        s.double().square().mean((1, 2)).sum(),
                        t.double().square().mean((1, 2)).sum(),
                    )
                )
            counts[index] += x.shape[0]

        hooks.append(native.register_forward_hook(measure, with_kwargs=True))
    try:
        for start in range(0, len(clips), 8):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                teacher(clips[start : start + 8], return_hidden=True)
    finally:
        for hook in hooks:
            hook.remove()
    records = []
    for i in indices:
        dot, square, target_square = moments[i].tolist()
        scale = dot / square
        if not 0 < scale < 4:
            raise ValueError(f"Unexpected fitted scale {scale} at layer {i}")
        student.body.layers[i].mixer.o_proj.weight.mul_(scale)
        records.append(
            {
                "layer": i,
                "examples": counts[i],
                "output_projection_scale": scale,
                "relative_mse_before": (square - 2 * dot + target_square) / target_square,
                "relative_mse_after_scale": (
                    scale * scale * square - 2 * scale * dot + target_square
                )
                / target_square,
            }
        )
    destination = Path(args.candidate)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite candidate {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".calibration-", dir=destination.parent))
    try:
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in student.body.state_dict().items()},
            str(temporary / "model.safetensors"),
        )
        student.body.config.save_pretrained(temporary)
        student.tokenizer.save_pretrained(temporary / "tokenizer")
        shutil.copy2("checkpoints/qwen35_0p8b_kda/kda_layout.json", temporary / "kda_layout.json")
        config = configuration(args.checkpoint, args.device)
        config["model"]["checkpoint"] = str(destination)
        config["checkpoint"] = str(destination)
        config["run_dir"] = "runs/streamnav_conversion_calibrated_candidate_20261008/ealm"
        (temporary / "resolved_config.yaml").write_text(yaml.safe_dump(config))
        provenance = {
            "protocol": __doc__,
            "candidate": str(destination),
            "source": "checkpoints/qwen35_0p8b_kda",
            "data": str(Path(args.data_root) / "train.pt"),
            "frames": len(clips),
            "records": records,
            "teacher": "Original pretrained native Qwen with official MRoPE",
            "not_promoted": True,
            "navigation_optimizer_included": False,
        }
        (temporary / "conversion_calibration.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    write_json(args.output, provenance)
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()

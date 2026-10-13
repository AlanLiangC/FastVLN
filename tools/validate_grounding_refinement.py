"""GPU check of real-data static resets, point replay, gradients and an optimizer update."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from streamnav.data.grounding import GroundingSource
from streamnav.models.point_cot.policy import Qwen35PointCoTPolicy
from streamnav.training.grounding_trainer import optimizer_for, validation_indices
from streamnav.training.point_supervision import grounding_loss, grounding_loss_options
from streamnav.training.pointcot_trainer import load_adapters
from streamnav.utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="runtime/reports/grounding_refinement_validation.json")
    args = parser.parse_args()
    config = yaml.safe_load(Path("configs/pointcot.yaml").read_text())
    config["model"]["gradient_checkpointing"] = False
    seed_everything(config["seed"])
    policy = Qwen35PointCoTPolicy(config).train()
    load_adapters(policy, args.checkpoint)
    source = GroundingSource(config["grounding"]["val_manifest"])
    rows = [GroundingSource.materialize(source.rows[i]) for i in validation_indices(source, 12)]
    options = grounding_loss_options(config["grounding"])
    optimizer = optimizer_for(policy, config["grounding"]["learning_rates"])
    point_before = policy.opos_rows.detach().clone()
    frozen_before = next(policy.native.model.visual.parameters()).detach().clone()
    with torch.no_grad():
        initial_loss, _, packed = grounding_loss(policy, rows, **options)
        features = GroundingSource.features_many(policy, rows)
        errors = []
        token_matches = []
        for i, (feature, row) in enumerate(zip(features, rows, strict=True)):
            step = policy.step(
                feature,
                policy.start_episode(row["image_id"], row["instruction"]),
                deterministic=True,
            )
            errors.append(float((step.opos_logits - packed.opos_logits[i]).abs().max()))
            token_matches.append(int(step.opos) == int(packed.opos_logits[i].argmax()))
        assert max(errors) < config["trainer"]["replay_tolerance"], errors
        assert all(token_matches), token_matches
    history = []
    for _ in range(6):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = grounding_loss(policy, rows, **options)
        history.append(float(loss.detach()))
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy._all_trainable, 1, error_if_nonfinite=True)
        assert all(p.grad is None for p in policy.parameters() if not p.requires_grad)
        optimizer.step()
    with torch.no_grad():
        after, _, _ = grounding_loss(policy, rows, **options)
        point_change = float((point_before - policy.opos_rows).abs().max())
        assert point_change > 0 and float(after) < float(initial_loss)
        assert torch.equal(frozen_before, next(policy.native.model.visual.parameters()))
    result = {
        "passed": True,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "real_query_count": len(rows),
        "sources": sorted({r["source"] for r in rows}),
        "packed_inference_opos_error_max": max(errors),
        "greedy_tokens_match": all(token_matches),
        "initial_loss": float(initial_loss),
        "after_six_updates_loss": float(after),
        "controlled_losses": history,
        "opos_parameter_change_max": point_change,
        "frozen_parameters_unchanged": True,
        "preclip_grad_norm": float(norm),
        "peak_vram_gib": torch.cuda.max_memory_allocated(policy.device) / 1024**3,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()

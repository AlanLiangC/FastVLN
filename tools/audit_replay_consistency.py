"""Measure PPO ratios BEFORE any optimizer step on real Habitat rollouts."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from streamnav.data.mixture import make_source
from streamnav.envs.habitat_client import VectorHabitatEnvs
from streamnav.training.checkpoint import load_policy
from streamnav.training.dagger import behavior_log_prob
from streamnav.training.rollout import RolloutCollector, replay_sequences
from streamnav.utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--modes", nargs="+", default=["recurrent", "chunk"])
    args = parser.parse_args()
    with initialize_config_dir(version_base=None, config_dir=str(Path("configs").resolve())):
        cfg = OmegaConf.to_container(compose(config_name="config"), resolve=True)
    saved = Path(args.checkpoint) / "resolved_config.yaml"
    if saved.exists():
        cfg = yaml.safe_load(saved.read_text())
    cfg.update(checkpoint=args.checkpoint, device=args.device)
    cfg["habitat"]["gpu_device_id"] = int(args.device.split(":")[-1])
    cfg["trainer"]["curriculum"]["enabled"] = False
    policy = load_policy(cfg, preserve_master_weights=True)
    cfg["trainer"]["rollout_steps"] = 8
    cfg["trainer"]["sequence_length"] = 4
    results = {}
    for mode in args.modes:
        seed_everything(17)
        policy.backbone.inference_mode = mode
        sources = [make_source(cfg["data"], 17 + i * 1009) for i in range(8)]
        envs = VectorHabitatEnvs(cfg["habitat"], 8)
        collector = RolloutCollector(policy, envs, sources, cfg["trainer"])
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                buffer = collector.collect(beta=0.16)
            differences, ratios, wrong, entropies = [], [], [], []
            policy.train()
            for seqs in buffer.sequence_batches(8, shuffle=True):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, values = replay_sequences(policy, buffer, seqs)
                    expert = buffer.gather("expert_actions", seqs, policy.backbone.device)
                    executed = buffer.gather("executed_actions", seqs, policy.backbone.device)
                    new = behavior_log_prob(logits, executed, expert, buffer.beta)
                    old = buffer.gather("old_log_probs", seqs, policy.backbone.device)
                    differences.append((new - old).detach().cpu().flatten())
                    ratios.append((new - old).exp().detach().cpu().flatten())
                    wrong.append((logits.argmax(-1) != expert).detach().cpu().flatten())
                    entropies.append(
                        torch.distributions.Categorical(logits=logits)
                        .entropy()
                        .detach()
                        .cpu()
                        .flatten()
                    )
                del logits, values, new, old
            diff, ratio = torch.cat(differences), torch.cat(ratios)
            entropy, incorrect = torch.cat(entropies), torch.cat(wrong)
            results[mode] = {
                "transitions": ratio.numel(),
                "before_optimizer": True,
                "log_prob_error_mean": diff.abs().mean().item(),
                "log_prob_error_max": diff.abs().max().item(),
                "ratio_min": ratio.min().item(),
                "ratio_max": ratio.max().item(),
                "ratio_outside_ppo_clip": ((ratio - 1).abs() > 0.2).float().mean().item(),
                "confident_wrong_alpha_zero_count": int(((entropy <= 0.2) & incorrect).sum()),
                "wrong_count": int(incorrect.sum()),
            }
            print(json.dumps({mode: results[mode]}), flush=True)
            del buffer, collector
        finally:
            envs.close()
            torch.cuda.empty_cache()
            Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

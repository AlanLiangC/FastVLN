"""Run the full 100-step reference batch and audit post-warmup DDP equality."""

import hashlib
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from streamnav.training import distributed as parallel
from streamnav.training.dagger import behavior_log_prob
from streamnav.training.rollout import replay_sequences
from streamnav.training.trainer import EndToEndObjectNavTrainer


class CheckedTrainer(EndToEndObjectNavTrainer):
    def update(self, buffer):
        try:
            metrics = super().update(buffer)
        except RuntimeError as error:
            if "before any optimizer step" not in str(error):
                raise
            # Diagnose batching independently of autograd/optimizer updates.
            comparisons = {}
            with torch.no_grad(), self.autocast():
                for batch_size in (buffer.num_envs, self.cfg["sequence_batch_size"]):
                    errors = []
                    for sequences in buffer.sequence_batches(batch_size, shuffle=False):
                        logits, _ = replay_sequences(self.policy, buffer, sequences)
                        actions = buffer.gather("executed_actions", sequences, self.device)
                        expert = buffer.gather("expert_actions", sequences, self.device)
                        new = behavior_log_prob(logits, actions, expert, buffer.beta)
                        old = buffer.gather("old_log_probs", sequences, self.device)
                        errors.append((new - old).abs().max().item())
                    comparisons[str(batch_size)] = max(errors)
            results = parallel.gather(comparisons)
            if parallel.rank() == 0:
                (self.run_dir / "replay_batch_diagnosis.json").write_text(
                    json.dumps(results, indent=2)
                )
            self.save_checkpoint()
            raise
        digest = hashlib.sha256()
        for parameter in self.policy.parameters():
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        hashes = parallel.gather(digest.hexdigest())
        if len(set(hashes)) != 1:
            raise AssertionError(f"DDP parameters diverged: {hashes}")
        if parallel.rank() == 0:
            print(
                json.dumps(
                    {
                        "ddp_all_parameters_equal": True,
                        "update": self.update_index + 1,
                        "parameter_sha256": hashes[0],
                    }
                ),
                flush=True,
            )
        return metrics


def main():
    import sys

    with initialize_config_dir(version_base=None, config_dir=str(Path("configs").resolve())):
        cfg = compose(config_name="config", overrides=sys.argv[1:])
    config = OmegaConf.to_container(cfg, resolve=True)
    parallel.initialize(config)
    try:
        CheckedTrainer(config).run()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

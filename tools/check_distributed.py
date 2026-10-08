"""Run with torchrun: compare distributed gradients with a full-batch reference."""

import json
import os

import torch
import torch.distributed as dist

from streamnav.training import distributed as parallel


class Fixture(torch.nn.Module):
    def __init__(self, model, extra, unused):
        super().__init__()
        self.model, self.extra, self.unused = model, extra, unused

    def forward(self, x, rank):
        loss = self.model(x).square().mean()
        return loss + self.extra.square() if rank == 0 else loss


def check(device, mode):
    rank, size = parallel.rank(), parallel.world_size()
    torch.manual_seed(31)
    model = torch.nn.Linear(3, 2, device=device)
    extra = torch.nn.Parameter(torch.tensor(2.0, device=device))
    unused = torch.nn.Parameter(torch.tensor(7.0, device=device))
    x = torch.arange(size * 12, device=device, dtype=torch.float32).reshape(size, 4, 3) / 13
    reference = torch.nn.Linear(3, 2, device=device)
    reference.load_state_dict(model.state_dict())
    reference_extra = torch.nn.Parameter(extra.detach().clone())
    fixture = Fixture(model, extra, unused)
    if mode == "ddp":
        fixture = torch.nn.parallel.DistributedDataParallel(
            fixture,
            device_ids=[torch.device(device).index],
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
    local_loss = fixture(x[rank], rank)
    local_loss.backward()
    if mode == "flat_allreduce":
        parallel.average_gradients([*model.parameters(), extra, unused])
    reference_loss = reference(x.reshape(-1, 3)).square().mean() + reference_extra.square() / size
    reference_loss.backward()
    errors = []
    for actual, expected in zip(
        [*model.parameters(), extra], [*reference.parameters(), reference_extra], strict=True
    ):
        torch.testing.assert_close(actual.grad, expected.grad, atol=1e-6, rtol=1e-5)
        errors.append((actual.grad - expected.grad).abs().max().item())
    assert unused.grad is None
    values = torch.arange(rank * 4, rank * 4 + 4, dtype=torch.float32)
    normalized = parallel.normalize_advantages(values, device)
    all_values = torch.arange(size * 4, dtype=torch.float32)
    expected = (values - all_values.mean()) / all_values.std(unbiased=False)
    torch.testing.assert_close(normalized, expected)
    assert parallel.any_rank(rank == size - 1, device)
    torch.optim.AdamW([*model.parameters(), extra], lr=0.01).step()
    params = torch.cat([p.detach().flatten() for p in [*model.parameters(), extra]])
    gathered = [torch.empty_like(params) for _ in range(size)]
    dist.all_gather(gathered, params)
    for p in gathered:
        torch.testing.assert_close(params, p, rtol=0, atol=0)
    if rank == 0:
        print(
            json.dumps(
                {
                    "world_size": size,
                    "mode": mode,
                    "max_gradient_error": max(errors),
                    "parameters_identical": True,
                    "global_advantages": "pass",
                    "global_stop": "pass",
                }
            )
        )


def main():
    config = {"distributed": {"gpu_offset": int(os.getenv("PROBE_GPU_OFFSET", "0"))}, "habitat": {}}
    parallel.initialize(config)
    for mode in ("flat_allreduce", "ddp"):
        check(config["device"], mode)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

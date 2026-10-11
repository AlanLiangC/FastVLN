import pytest
import torch
from torch import nn

from streamnav.evaluation.perception_records import perception_records
from streamnav.models.lightnav_kda.adapters import LoRALinear
from streamnav.models.lightnav_kda.policy import TokenStep, video_position_ids
from streamnav.training.lightnav_trainer import (
    advantages,
    point_kl,
    replay_losses,
    resume_recipe,
    token_log_prob,
)


def test_joint_policy_probability_uses_both_generated_points():
    logits = [torch.tensor([0.0, 2.0]), torch.tensor([1.0, -1.0]), torch.tensor([0.0, 0.0])]
    tokens = [torch.tensor(1), torch.tensor(0), torch.tensor(1)]
    joint = token_log_prob(logits, tokens)
    expected = sum(x.log_softmax(-1)[y] for x, y in zip(logits, tokens, strict=True))
    assert joint == pytest.approx(expected.item())
    changed = token_log_prob(logits, [torch.tensor(0), *tokens[1:]])
    assert changed != joint


def test_gae_does_not_cross_episode_and_bootstraps_timeout():
    reward = torch.tensor([[1.0], [10.0]])
    values = torch.zeros_like(reward)
    dones = torch.tensor([[True], [True]])
    a, returns = advantages(
        reward,
        values,
        dones,
        torch.tensor([100.0]),
        gamma=0.5,
        tau=1.0,
        timeout_values=torch.tensor([[2.0], [0.0]]),
    )
    assert a.tolist() == [[2.0], [10.0]]
    assert torch.equal(a, returns)


def test_point_retention_normalizes_half_precision_teacher():
    probabilities = torch.tensor([0.1, 0.3, 0.6], dtype=torch.float16)
    normalized = probabilities.float() / probabilities.sum(dtype=torch.float32)
    assert point_kl(normalized.log(), probabilities).item() == pytest.approx(0.0, abs=1e-6)


def test_low_rank_increment_preserves_frozen_base_and_learns():
    base = nn.Linear(4, 3)
    layer = LoRALinear(base, rank=2, alpha=2)
    x = torch.randn(5, 4)
    assert torch.equal(layer(x), base(x))
    layer(x).square().mean().backward()
    assert base.weight.grad is None
    assert layer.b.grad.abs().sum() > 0
    with torch.no_grad():
        layer.b.add_(0.1)
    layer.enabled = False
    assert torch.equal(layer(x), base(x))


def test_point_cot_video_records_do_not_invent_arrival_head():
    apos, opos = torch.zeros(1, 1300), torch.zeros(1, 1297)
    apos[0, 130] = opos[0, 250] = 10
    record = perception_records({"apos": apos, "opos": opos, "point_cot": True})[0]
    assert record["point_cot"]
    assert record["apos"] == 130 and record["opos"] == 250
    assert "arrival_probabilities" not in record


def test_resume_recipe_allows_more_updates_but_rejects_changed_ppo():
    import copy

    import yaml

    config = yaml.safe_load(open("configs/lightnav_kda.yaml"))
    longer = copy.deepcopy(config)
    longer["trainer"]["num_updates"] += 1
    longer["habitat"]["gpu_device_id"] = 7
    longer["trainer"]["replay_mode"] = "sequential"
    longer["model"]["language_compute_dtype"] = "bfloat16"
    longer["model"]["float32_matmul_precision"] = "highest"
    longer["model"]["recurrent_decode"] = False
    assert resume_recipe(config) == resume_recipe(longer)
    longer["trainer"]["ppo_weight"] = 0
    assert resume_recipe(config) != resume_recipe(longer)


def test_initial_and_best_checkpoints_survive_retention(tmp_path):
    import json

    from streamnav.training.lightnav_trainer import record_best, save

    class Policy:
        def trainable_state(self):
            return {"adapter": torch.ones(2)}

        def parameter_summary(self):
            return {"source_revision": {"revision": "test"}}

    class Source:
        def state_dict(self):
            return {}

    optimizer = torch.optim.AdamW([nn.Parameter(torch.ones(2))])
    cfg = {"run_dir": str(tmp_path), "trainer": {"keep_checkpoints": 2}}
    for update in range(6):
        save(Policy(), optimizer, Source(), cfg, update, frames=10 * update)
        if update == 1:
            record_best(cfg, update, {"seen": {"success": 0.5}})
    assert (tmp_path / "checkpoints/best").resolve().name == "update_0000001"
    assert (tmp_path / "checkpoints/update_0000000").exists()
    assert not (tmp_path / "checkpoints/update_0000002").exists()
    assert json.loads((tmp_path / "latest_checkpoint.json").read_text())["update"] == 5


def test_timestamp_separated_video_tubelets_receive_valid_native_rope():
    from types import MethodType, SimpleNamespace

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    model = SimpleNamespace(
        config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
    )
    model.get_vision_position_ids = MethodType(Qwen3VLModel.get_vision_position_ids, model)
    ids = torch.tensor([[9, 7, 7, 7, 7, 8, 7, 7, 7, 7, 9]])
    types = torch.where(ids == 7, 2, 0)
    grid = torch.tensor([[2, 4, 4]])

    def native(**kwargs):
        positions, _ = Qwen3VLModel.get_rope_index(model, **kwargs)
        return positions

    positions = video_position_ids(
        native, input_ids=ids, mm_token_type_ids=types, video_grid_thw=grid
    )
    assert positions.shape == (3, 1, 11)
    assert positions[:, 0, 6].min() > positions[:, 0, 1].max()
    assert grid.tolist() == [[2, 4, 4]]  # ViT still consumes the unsplit temporal grid.


def test_zero_mean_ppo_loss_still_has_policy_gradient():
    from streamnav.training.ppo import clipped_ppo_loss

    logits = torch.zeros(2, requires_grad=True)
    log_probs = logits.log_softmax(-1)[torch.tensor([0, 1])]
    loss, _ = clipped_ppo_loss(log_probs, log_probs.detach(), torch.tensor([1.0, -1.0]))
    loss = loss.mean()
    assert loss.item() == 0
    loss.backward()
    assert logits.grad.abs().sum().item() > 0


def test_batched_ppo_trains_all_three_tokens_only_on_student_transitions():
    import yaml

    logits = [torch.zeros(2, 3, requires_grad=True) for _ in range(3)]
    tokens = torch.tensor([0, 1])
    step = TokenStep(tokens, tokens, tokens, *logits, torch.zeros(2), None)
    old = step.log_prob.detach()
    records = [
        {"expert": 0, "student_action": i == 0, "old_log_prob": old[i], "teacher": None}
        for i in range(2)
    ]
    cfg = yaml.safe_load(open("configs/lightnav_kda.yaml"))["trainer"]
    _, components, error = replay_losses(
        step, records, torch.tensor([1.0, -1.0]), torch.zeros(2), cfg, 0.5
    )
    assert error.item() == 0
    assert components["point_kl"].tolist() == [0.0, 0.0]
    components["ppo_loss"].sum().backward()
    for x in logits:
        assert x.grad[0].abs().sum() > 0
        assert x.grad[1].abs().sum() == 0

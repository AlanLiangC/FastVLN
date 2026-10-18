import pytest
import torch

from streamnav.evaluation.perception_records import perception_records
from streamnav.models.point_cot.planning import FutureActionPlan
from streamnav.training.action_planning import attach_future_actions, future_action_loss
from streamnav.training.optimizer_migration import load_optimizer_with_added_groups
from streamnav.training.point_supervision import forward_intent_loss


def record(action, *, expert=True, episode="a", done=False):
    return {
        "expert_stream": expert,
        "expert": action,
        "episode_id": episode,
        "tokens": (torch.tensor(0), torch.tensor(0), torch.tensor(action)),
        "done": done,
    }


def test_future_labels_stop_at_terminal_and_never_supervise_student_future():
    rows = [
        [record(a, episode=e, done=d), record(3, expert=False)]
        for a, e, d in ((1, "a", False), (3, "a", False), (0, "a", True), (2, "a", False))
    ]
    attach_future_actions(rows)
    assert rows[0][0]["future_actions"] == [1, 3, 0, -100]
    assert rows[1][0]["future_actions"] == [3, 0, -100, -100]
    assert rows[2][0]["future_actions"] == [0, -100, -100, -100]
    assert rows[3][0]["future_actions"] == [2, -100, -100, -100]
    assert all(row[1]["future_actions"] == [-100] * 4 for row in rows)
    rows[1][0]["tokens"] = (torch.tensor(0), torch.tensor(0), torch.tensor(2))
    with pytest.raises(ValueError, match="expert-executed"):
        attach_future_actions(rows)


def test_future_labels_do_not_cross_episode_or_rollout_boundary():
    rows = [[record(1)], [record(2, episode="b")]]
    attach_future_actions(rows)
    assert rows[0][0]["future_actions"] == [1, -100, -100, -100]


def test_future_loss_masks_unexecuted_horizons_and_current_action():
    logits = torch.zeros(2, 4, 6, requires_grad=True)
    loss = future_action_loss(
        logits, [{"future_actions": [1, 3, 0, -100]}, {"future_actions": [-100] * 4}]
    )
    loss.sum().backward()
    assert logits.grad[:, 0].count_nonzero() == 0
    assert logits.grad[0, 1, 3] < 0 and logits.grad[0, 2, 0] < 0
    assert logits.grad[0, 3].count_nonzero() == 0
    assert logits.grad[1].count_nonzero() == 0


def test_case_plan_stops_at_predicted_stop_and_preserves_executed_point_prefix():
    plans = torch.zeros(2, 4, 6)
    tokens = torch.tensor([[1, 0, 3, 1], [2, 1, 3, 1]])
    plans.scatter_(-1, tokens.unsqueeze(-1), 5.0)
    predictions = {
        "point_cot": True,
        "action_plan_logits": plans,
        "apos": torch.zeros(2, 1300),
        "opos": torch.zeros(2, 1300),
        "apos_tokens": torch.tensor([42, 1297]),
        "opos_tokens": torch.tensor([11, 0]),
    }
    records = perception_records(predictions)
    assert records[0]["action_plan"] == [1, 0]
    assert len(records[0]["action_plan_probabilities"]) == 2
    assert records[1]["action_plan"] == [2, 1, 3, 1]
    assert records[0]["apos"] == 42 and records[1]["apos"] == 1297
    assert records[0]["opos"] == 11


def test_missing_pixel_supervision_keeps_forward_compatibility_without_invented_point():
    logits = torch.zeros(3, 1300, requires_grad=True)
    labels = [{"apos_targets": []}, {"apos_targets": [42]}, {"apos_targets": []}]
    loss, mask = forward_intent_loss(logits, labels, torch.tensor([1, 2, 3]))
    assert mask.tolist() == [True, False, False]
    loss.sum().backward()
    assert logits.grad[0, 1297] > 0 and logits.grad[0, 1] < 0 and logits.grad[0, 0] < 0
    assert logits.grad[1:].count_nonzero() == 0
    assert labels[0]["apos_targets"] == []


def test_planning_branch_has_distinct_horizon_gradients():
    module = FutureActionPlan(16, 3, bottleneck=8)
    hidden = torch.randn(2, 16, requires_grad=True)
    assert module(hidden).shape == (2, 3, 16)
    torch.testing.assert_close(module(hidden), hidden[:, None].expand(-1, 3, -1))
    (module(hidden) * torch.randn(2, 3, 16)).sum().backward()
    assert all(module.up.weight.grad.reshape(3, 16, 8)[i].norm() > 0 for i in range(3))
    assert hidden.grad.norm() > 0


@pytest.mark.parametrize("new_group", ["planning", "hint"])
def test_added_optimizer_group_preserves_adam_moments_by_group(new_group):
    original = torch.nn.Parameter(torch.ones(3))
    old = torch.optim.AdamW([{"params": [original], "name": "action"}], lr=0.1)
    original.grad = torch.tensor([1.0, 2.0, 3.0])
    old.step()
    saved = old.state_dict()
    plan, current = torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(3))
    new = torch.optim.AdamW(
        [{"params": [plan], "name": new_group}, {"params": [current], "name": "action"}], lr=0.02
    )
    load_optimizer_with_added_groups(new, saved)
    torch.testing.assert_close(new.state[current]["exp_avg"], old.state[original]["exp_avg"])
    assert not new.state[plan]
    assert new.state[current]["step"] == old.state[original]["step"]

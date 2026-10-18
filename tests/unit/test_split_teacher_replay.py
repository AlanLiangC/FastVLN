import pytest
import torch

from streamnav.models.point_cot.policy import TokenStep
from streamnav.training.pointcot_trainer import replay_losses, teacher_fragment_losses


@pytest.mark.parametrize("mix", [0.0, 0.5, 1.0])
def test_separate_backward_preserves_all_losses_and_shared_parameter_gradients(mix):
    generator = torch.Generator().manual_seed(71)
    parameter = torch.tensor(0.3, requires_grad=True)
    templates = [torch.randn(3, n, generator=generator) for n in (1300, 1297, 6, 18)]
    teacher_templates = [torch.randn(3, n, generator=generator) for n in (1300, 1297, 6, 18)]
    tokens = torch.tensor([0, 1, 2])

    def step(bases):
        apos, opos, action, future = [x * parameter for x in bases]
        plan = torch.cat((action[:, None], future.view(3, 3, 6)), dim=1)
        return TokenStep(tokens, tokens, tokens, apos, opos, action, action[:, 0], None, plan)

    class Policy:
        device = torch.device("cpu")

        def replay_fragment(self, features, state, forced, resets=None):
            assert [int(t[0]) for t in forced] == [1299, 42, 2]
            return step(teacher_templates)

    old = step(templates).log_prob.detach()
    labels = [
        {
            "apos_valid": True,
            "apos": 1299,
            "opos_valid": True,
            "opos": 42,
            "opos_targets": [42, 43],
        },
        {"apos_valid": True, "apos": 42, "opos_valid": True, "opos": 0},
        {},
    ]
    records = [
        {
            "features": None,
            "tokens": (t, t, t),
            "expert": i,
            "perception_label": labels[i],
            "old_log_prob": old[i],
            "student_action": i == 0,
            "il_weight": i + 1,
            "future_actions": [i, 1, -100, 0],
        }
        for i, t in enumerate(tokens)
    ]
    cfg = {
        "teacher_action_mix": mix,
        "point_supervision_prefix": "actual",
        "num_envs": 3,
        "num_expert_envs": 1,
        "num_dagger_envs": 1,
        "ppo_clip": 0.2,
        "point_il_coef": 0.25,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "future_action_loss_coef": 0.25,
        "opos_supervision": {"anchor_weight": 0.2},
    }
    adv, returns = torch.tensor([1.0, -0.4, 0.7]), torch.ones(3)
    combined, combined_metrics, _ = replay_losses(
        Policy(),
        None,
        {},
        step(templates),
        records,
        adv,
        returns,
        cfg,
        0.25,
    )
    expected = {k: v.detach() for k, v in combined_metrics.items()}
    reference_gradient = torch.autograd.grad(combined.mean(), parameter)[0]
    actual, actual_metrics, _ = replay_losses(
        Policy(),
        None,
        {},
        step(templates),
        records,
        adv,
        returns,
        cfg,
        0.25,
        include_teacher=False,
    )
    actual.mean().backward()
    teacher, teacher_metrics = teacher_fragment_losses(Policy(), None, {}, records, cfg)
    teacher.mean().backward()
    torch.testing.assert_close(actual.detach() + teacher.detach(), combined.detach())
    torch.testing.assert_close(parameter.grad, reference_gradient)
    for key, value in expected.items():
        torch.testing.assert_close(
            actual_metrics[key].detach() + teacher_metrics.get(key, 0), value
        )


def test_teacher_prefix_point_loss_cannot_be_silently_excluded():
    tokens = torch.zeros(1, dtype=torch.long)
    logits = torch.zeros(1, 6, requires_grad=True)
    step = TokenStep(tokens, tokens, tokens, logits, logits, logits, logits[:, 0], None)
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="actual-history"):
        replay_losses(
            SimpleNamespace(device="cpu"),
            None,
            {},
            step,
            [{"expert": 0, "features": None, "tokens": (tokens[0],) * 3}],
            None,
            None,
            {"point_supervision_prefix": "teacher"},
            0.25,
            include_teacher=False,
        )

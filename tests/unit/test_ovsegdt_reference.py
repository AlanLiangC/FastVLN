"""Differential checks execute the local upstream math, not copied expectations."""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from streamnav.models.policy.actor_critic import NavigationActorCritic
from streamnav.training.ealm import EntropyAdaptiveLossMixer
from streamnav.training.optimizer import OVSegDTLRScheduler, build_optimizer
from streamnav.training.ppo import ovsegdt_value_loss

REFERENCE = Path("third_party/OVSegDT/ovon")
pytestmark = pytest.mark.skipif(not REFERENCE.exists(), reason="Local OVSegDT reference required")


def execute(nodes, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "upstream", "exec"), namespace)


def test_ealm_matches_reference_across_batches_and_resume():
    tree = ast.parse((REFERENCE / "algos/dagger_ppo.py").read_text())
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_update_ppo_ratio"
    )
    namespace = {}
    execute([method], namespace)
    reference = SimpleNamespace(entropy_ema=None, entropy_low=0.35, entropy_high=0.75)
    mixer = EntropyAdaptiveLossMixer()
    for index, entropy in enumerate([1.7] * 10 + [0.0] * 70 + [0.55] * 20 + [1.7] * 20):
        namespace["_update_ppo_ratio"](reference)
        _, alpha = mixer(torch.ones(6), torch.zeros(6), torch.rand(6))
        torch.testing.assert_close(alpha, torch.full((6,), 1 - reference.ppo_ratio))
        mixer.observe_entropy(entropy)
        reference.entropy_ema = (
            entropy
            if reference.entropy_ema is None
            else 0.95 * reference.entropy_ema + 0.05 * entropy
        )
        if index == 40:
            restored = EntropyAdaptiveLossMixer()
            restored.load_state_dict(mixer.state_dict())
            mixer = restored


def test_value_clipping_matches_upstream_values_and_gradients():
    tree = ast.parse((REFERENCE / "algos/dagger_ppo.py").read_text())
    statements = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "self.use_clipped_value_loss":
            statements.append(node)
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "value_loss" for t in node.targets
        ):
            if "F.mse_loss" in ast.unparse(node.value):
                statements.append(node)
    statements.sort(key=lambda n: n.lineno)
    assert len(statements) == 2
    for clip in [False, True]:
        values = torch.tensor([-0.3, -0.2, -0.19, 0.19, 0.2, 0.3], requires_grad=True)
        expected = values.detach().clone().requires_grad_(True)
        returns = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        namespace = {
            "torch": torch,
            "F": nn.functional,
            "values": expected,
            "batch": {"value_preds": torch.zeros(6), "returns": returns},
            "self": SimpleNamespace(use_clipped_value_loss=clip, clip_param=0.2),
        }
        execute(statements, namespace)
        loss = ovsegdt_value_loss(values, torch.zeros(6), returns, 0.2, clip).mean()
        torch.testing.assert_close(loss, namespace["value_loss"])
        loss.backward()
        namespace["value_loss"].backward()
        torch.testing.assert_close(values.grad, expected.grad)


def test_optimizer_schedule_matches_actual_pirlnav_class():
    tree = ast.parse((REFERENCE / "utils/lr_scheduler.py").read_text())
    namespace = {"logger": logging.getLogger("reference_test")}
    execute([n for n in tree.body if isinstance(n, ast.ClassDef)], namespace)
    policy = nn.Module()
    policy.backbone = nn.Module()
    policy.backbone.vision = nn.Linear(4, 4).requires_grad_(False)
    policy.backbone.state = nn.Linear(4, 4)
    policy.actor_critic = NavigationActorCritic(4)
    config = dict(
        head_lr=2.5e-4,
        backbone_lr=2.5e-4,
        vision_lr=0.0,
        weight_decay=0.0,
        optimizer_eps=1e-5,
        actor_warmup_updates=1,
    )
    optimizer = build_optimizer(policy, config)
    scheduler = OVSegDTLRScheduler(optimizer, config)
    assert type(optimizer) is torch.optim.Adam
    assert all(g["eps"] == 1e-5 and g["weight_decay"] == 0 for g in optimizer.param_groups)
    assert not any(p.requires_grad for p in policy.backbone.parameters())
    upstream_optimizer = torch.optim.Adam(
        [{"params": [nn.Parameter(torch.ones(1))], "lr": lr} for lr in [2.5e-4, 0.0, 0.0]]
    )
    reference_agent = SimpleNamespace(
        parameters=lambda: [],
        optimizer=upstream_optimizer,
        actor_critic=SimpleNamespace(
            unfreeze_actor=lambda: None,
            unfreeze_state_encoder=lambda: None,
            unfreeze_new_params=lambda: None,
        ),
    )
    reference = namespace["PIRLNavLRScheduler"](
        upstream_optimizer, reference_agent, 312500, 2.5e-4, 2.5e-4, 1e-5, 1, 1, 1, 1
    )
    for _ in range(5):
        assert scheduler.get_last_lr()[:3] == [g["lr"] for g in upstream_optimizer.param_groups]
        scheduler.step()
        reference.step()
        assert all(p.requires_grad for p in policy.backbone.state.parameters())
        assert not any(p.requires_grad for p in policy.backbone.vision.parameters())
    restored = OVSegDTLRScheduler(optimizer, config)
    restored.load_state_dict(scheduler.state_dict())
    assert restored.get_last_lr() == scheduler.get_last_lr()

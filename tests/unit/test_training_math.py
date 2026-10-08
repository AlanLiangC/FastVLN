import math

import pytest
import torch

from streamnav.contracts.action import NavigationAction
from streamnav.models.policy.action_distribution import ObjectNavActionDistribution
from streamnav.models.policy.actor_critic import NavigationActorCritic
from streamnav.training.dagger import DaggerBetaScheduler, behavior_log_prob, select_env_action
from streamnav.training.ealm import EntropyAdaptiveLossMixer
from streamnav.training.gae import compute_gae
from streamnav.training.optimizer import gradient_norm
from streamnav.training.ppo import clipped_ppo_loss
from streamnav.training.rewards import ObjectNavReward


def test_action_contract():
    assert [(a.name, int(a)) for a in NavigationAction] == [
        ("STOP", 0),
        ("MOVE_FORWARD", 1),
        ("TURN_LEFT", 2),
        ("TURN_RIGHT", 3),
    ]
    with pytest.raises(ValueError):
        NavigationAction(4)


def test_actor_critic_distribution():
    head = NavigationActorCritic(16, 8)
    logits, values = head(torch.randn(3, 16))
    assert logits.shape == (3, 4) and values.shape == (3,)
    dist = ObjectNavActionDistribution().build(torch.zeros(3, 4))
    torch.testing.assert_close(dist.entropy(), torch.full((3,), math.log(4)))
    with pytest.raises(ValueError):
        ObjectNavActionDistribution().build(torch.zeros(3, 5))


def test_gae_terminal_prevents_cross_episode_leakage():
    rewards = torch.tensor([[1.0], [100.0]])
    values = torch.zeros_like(rewards)
    advantages, returns = compute_gae(
        rewards, values, torch.tensor([[True], [True]]), torch.tensor([999.0]), 1, 1
    )
    torch.testing.assert_close(returns, rewards)
    torch.testing.assert_close(advantages, rewards)


def test_gae_bootstrap_and_timeout():
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[0.5], [1.0]])
    _, returns = compute_gae(
        rewards, values, torch.zeros(2, 1, dtype=torch.bool), torch.tensor([3.0]), 1, 1
    )
    torch.testing.assert_close(returns, torch.tensor([[6.0], [5.0]]))
    _, returns = compute_gae(
        torch.tensor([[1.0]]),
        torch.tensor([[0.5]]),
        torch.tensor([[True]]),
        torch.tensor([1000.0]),
        0.9,
        1,
        torch.tensor([[2.0]]),
    )
    torch.testing.assert_close(returns, torch.tensor([[2.8]]))


def test_ppo_clips_opposite_advantage_signs():
    loss, ratio = clipped_ppo_loss(
        torch.tensor([math.log(1.5), math.log(0.5)]), torch.zeros(2), torch.tensor([1.0, -1.0])
    )
    torch.testing.assert_close(loss, torch.tensor([-1.2, 0.8]))
    torch.testing.assert_close(ratio, torch.tensor([1.5, 0.5]))


def test_dagger_schedule_and_action_extremes():
    schedule = DaggerBetaScheduler(0.8, 0.05, 10)
    assert schedule.beta == 0.8
    for _ in range(30):
        schedule.step()
    assert schedule.beta == pytest.approx(0.05)
    policy, expert = torch.zeros(50, dtype=torch.long), torch.ones(50, dtype=torch.long)
    assert torch.equal(select_env_action(policy, expert, 0)[0], policy)
    assert torch.equal(select_env_action(policy, expert, 1)[0], expert)
    with pytest.raises(ValueError):
        DaggerBetaScheduler(decay_updates=0)


def test_dagger_ppo_uses_executed_behavior_probability():
    logits = torch.zeros(2, 4, requires_grad=True)
    executed, expert = torch.tensor([1, 2]), torch.tensor([1, 1])
    probs = behavior_log_prob(logits, executed, expert, 0.8).exp()
    torch.testing.assert_close(probs, torch.tensor([0.85, 0.05]))
    assert not torch.allclose(probs, torch.full((2,), 0.25))
    (-probs.log().sum()).backward()
    assert logits.grad.abs().sum() > 0


def test_ealm_entropy_weights_detached_and_monotonic():
    entropy = torch.tensor([0.0, 0.7, 1.4], requires_grad=True)
    il = torch.ones(3, requires_grad=True)
    rl = torch.zeros(3, requires_grad=True)
    loss, alpha = EntropyAdaptiveLossMixer()(il, rl, entropy)
    torch.testing.assert_close(alpha, torch.tensor([0.0, 0.5, 1.0]))
    loss.sum().backward()
    assert entropy.grad is None
    fixed, _ = EntropyAdaptiveLossMixer(enabled=False, fixed_alpha=0.25)(il, rl, entropy)
    torch.testing.assert_close(fixed, torch.full((3,), 0.25))


def test_reward_success_and_bad_distance():
    reward = ObjectNavReward()
    assert reward.compute(2, 1, False, False) == pytest.approx(0.99)
    assert reward.compute(0, 0, True, False, True) == pytest.approx(9.99)
    with pytest.raises(ValueError):
        reward.compute(float("inf"), 0, False, False)


def test_foreach_gradient_norm_matches_flattened_gradients():
    parameters = [
        torch.nn.Parameter(torch.zeros(2)),
        torch.nn.Parameter(torch.zeros(3)),
        torch.nn.Parameter(torch.zeros(1)),
    ]
    parameters[0].grad = torch.tensor([3.0, 4.0])
    parameters[1].grad = torch.tensor([0.0, 0.0, 12.0])
    assert gradient_norm(iter(parameters)) == pytest.approx(13.0)
    assert gradient_norm([parameters[2]]) == 0

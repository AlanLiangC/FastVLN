from types import SimpleNamespace

import torch

from streamnav.contracts.action import NavigationAction
from streamnav.contracts.state import LayerState, PolicyOutput, StreamingState
from streamnav.serving.session import SessionManager


class TestPolicy:
    __test__ = False

    def eval(self):
        return self

    def start_episode(self, episode, instruction):
        return StreamingState(
            (LayerState(None, torch.zeros(1, 1)),), episode, instruction, 0, instruction
        )

    reset = start_episode

    def act(self, rgb, state, deterministic):
        return NavigationAction.MOVE_FORWARD, PolicyOutput(
            torch.tensor([0.0, 1.0, 0.0, 0.0]), torch.tensor(1.0), state.with_cache(state.kda_cache)
        )


def test_episode_reset_and_session_isolation():
    manager = SessionManager(TestPolicy(), max_sessions=2)
    a, b = manager.create("chair"), manager.create("bed")
    manager.step(a, torch.zeros(2, 2, 3))
    assert manager.sessions[a].state.step_index == 1
    assert manager.sessions[b].state.step_index == 0
    manager.reset(a, "plant")
    assert manager.sessions[a].state.step_index == 0
    assert manager.sessions[a].state.instruction == "plant"
    manager.close(a)
    assert a not in manager.sessions


def test_checkpoint_replacement_invalidates_old_recurrent_states():
    manager = SessionManager(TestPolicy())
    session = manager.create("chair")
    replacement = TestPolicy()
    manager.replace_policy(replacement)
    assert manager.policy is replacement
    assert session not in manager.sessions
    fresh = manager.create("chair")
    assert manager.sessions[fresh].state.step_index == 0


def test_serving_uses_evaluation_autocast_for_prefill_step_reset_and_batch():
    from streamnav.serving.batching import batch_steps

    class MixedPrecisionPolicy(TestPolicy):
        inference_compute_dtype = torch.bfloat16
        backbone = SimpleNamespace(device=torch.device("cpu"))

        def start_episode(self, *args):
            assert torch.is_autocast_enabled("cpu")
            assert torch.get_autocast_dtype("cpu") == torch.bfloat16
            return super().start_episode(*args)

        reset = start_episode

        def act(self, *args, **kwargs):
            assert torch.is_autocast_enabled("cpu")
            return super().act(*args, **kwargs)

        def forward_batch(self, rgb, states):
            assert torch.is_autocast_enabled("cpu")
            return torch.zeros(len(states), 4), torch.zeros(len(states)), states

    manager = SessionManager(MixedPrecisionPolicy())
    session = manager.create("chair")
    rgb = torch.zeros(2, 2, 3)
    manager.step(session, rgb)
    manager.reset(session, "bed")
    assert len(batch_steps(manager, [(session, rgb)])) == 1

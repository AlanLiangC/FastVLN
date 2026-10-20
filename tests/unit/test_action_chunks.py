import pytest

from streamnav.evaluation.action_chunks import ActionChunk


def test_committed_forward_survives_a_new_opposite_turn_prediction():
    chunk = ActionChunk(4)
    chunk.start([2, 2, 1, 1], 34)
    chunk.advance(2)
    assert chunk.forced_action == 2
    with pytest.raises(ValueError, match="unfinished"):
        chunk.start([3, 1, 1, 1], 35)
    chunk.advance(2)
    assert chunk.forced_action == 1
    chunk.advance(1)
    chunk.advance(1)
    assert chunk.forced_action is None
    assert chunk.reason == "horizon_complete"


def test_single_step_is_always_replanned():
    chunk = ActionChunk(1)
    chunk.start([2, 1, 1, 1], 0)
    chunk.advance(2)
    assert chunk.forced_action is None
    chunk.start([3, 1, 1, 1], 1)
    assert chunk.forced_action == 3


@pytest.mark.parametrize(
    "collision,done,reason", [(True, False, "collision"), (False, True, "episode_end")]
)
def test_collision_and_terminal_discard_remaining_actions(collision, done, reason):
    chunk = ActionChunk(4)
    chunk.start([1, 1, 2, 1], 0)
    chunk.advance(1, collision=collision, done=done)
    assert chunk.forced_action is None and chunk.reason == reason


def test_stop_truncation_and_executed_action_check():
    chunk = ActionChunk(4)
    chunk.start([1, 0, 2, 3], 0)
    assert chunk.plan == (1, 0)
    with pytest.raises(ValueError, match="differs"):
        chunk.advance(2)
    chunk.advance(1)
    assert chunk.forced_action == 0
    chunk.advance(0, done=True)
    assert chunk.forced_action is None

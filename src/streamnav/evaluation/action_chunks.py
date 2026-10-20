"""Experimental fixed-horizon execution; independent of the production policy."""

from dataclasses import dataclass


@dataclass
class ActionChunk:
    horizon: int
    plan: tuple[int, ...] = ()
    offset: int = 0
    origin_frame: int = -1
    reason: str = "episode_start"

    def __post_init__(self):
        if self.horizon not in (1, 2, 4):
            raise ValueError("Execution horizon must be 1, 2 or 4")

    @property
    def forced_action(self):
        return self.plan[self.offset] if self.offset < len(self.plan) else None

    def start(self, predicted, frame):
        if self.forced_action is not None:
            raise ValueError("Cannot replace an unfinished action chunk")
        plan = tuple(int(a) for a in predicted[: self.horizon])
        if not plan or any(a not in range(6) for a in plan):
            raise ValueError("Invalid action plan")
        if 0 in plan:
            plan = plan[: plan.index(0) + 1]
        self.plan, self.offset, self.origin_frame = plan, 0, frame

    def advance(self, executed, *, collision=False, done=False):
        if self.forced_action is None or int(executed) != self.forced_action:
            raise ValueError("Executed action differs from the committed plan")
        self.offset += 1
        if done or collision:
            self.plan, self.offset = (), 0
            self.reason = "episode_end" if done else "collision"
        elif self.offset == len(self.plan):
            self.reason = "horizon_complete"

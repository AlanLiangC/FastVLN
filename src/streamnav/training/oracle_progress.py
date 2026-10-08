"""Attribute follower stalls only to actions that actually follow its advice."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OracleProgressTracker:
    best_distance: float
    patience: int = 64
    min_progress: float = 0.02
    stagnant_teacher_steps: int = 0
    last_advice: int | None = None

    def record_advice(self, action):
        self.last_advice = int(action)

    def record_step(self, action, distance):
        followed = self.last_advice is not None and int(action) == self.last_advice
        self.last_advice = None
        if not followed or distance < self.best_distance - self.min_progress:
            self.best_distance = distance
            self.stagnant_teacher_steps = 0
        else:
            self.stagnant_teacher_steps += 1

    @property
    def stalled(self):
        return self.stagnant_teacher_steps >= self.patience

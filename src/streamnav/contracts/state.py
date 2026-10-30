from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class LayerState:
    conv: torch.Tensor | None
    recurrent: torch.Tensor | None


@dataclass(frozen=True)
class StreamingState:
    kda_cache: tuple[LayerState, ...]
    episode_id: str
    instruction_hash: str
    step_index: int
    instruction: str
    next_action_probabilities: torch.Tensor | None = None
    previous_action: int = -1
    previous_collision: bool = False
    previous_displacement: float | None = None
    stationary_steps: int = 0
    alternating_steps: int = 0
    frame_signature: tuple[float, ...] | None = None
    visual_stationary_steps: int = 0
    predicted_action: int = -1

    def with_cache(self, cache, *, next_action_probabilities=None):
        return replace(
            self,
            kda_cache=cache,
            step_index=self.step_index + 1,
            next_action_probabilities=next_action_probabilities,
        )

    def after_action(self, collision=False, *, action=None, displacement=None):
        if action is None and self.predicted_action >= 0:
            action = self.predicted_action
        if action is None:
            return replace(self, next_action_probabilities=None) if collision else self
        action = int(action)
        alternating = (
            self.alternating_steps + 1
            if action in (2, 3)
            and self.previous_action in (2, 3)
            and action != self.previous_action
            else int(action in (2, 3))
        )
        return replace(
            self,
            next_action_probabilities=None if collision else self.next_action_probabilities,
            previous_action=action,
            previous_collision=bool(collision),
            previous_displacement=displacement,
            stationary_steps=self.stationary_steps + 1
            if displacement is not None and displacement < 0.0125
            else 0,
            alternating_steps=alternating,
        )

    def motion_feedback(self, visual_change=None):
        """Only executed actions and RGB changes; no simulator collision or pose."""
        stagnant = visual_change is not None and visual_change < 0.01
        previous = self.previous_action if self.previous_action >= 0 else self.predicted_action
        return (
            *(float(previous == a) for a in range(7)),
            float(stagnant and previous == 1),
            min(2.0, max(0.0, visual_change or 0.0) * 10),
            float(visual_change is not None),
            min(self.visual_stationary_steps + 1, 32) / 32 if stagnant else 0.0,
            min(self.alternating_steps, 32) / 32,
        )


@dataclass
class PolicyOutput:
    logits: torch.Tensor
    value: torch.Tensor
    state: StreamingState

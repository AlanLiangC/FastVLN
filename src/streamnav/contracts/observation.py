from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Observation:
    rgb: torch.Tensor  # uint8 HWC, RGB only; no privileged simulator inputs
    frame_id: int
    timestamp_s: float

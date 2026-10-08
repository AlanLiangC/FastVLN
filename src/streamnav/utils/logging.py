import json
import warnings
from collections import deque
from pathlib import Path

import torch


def append_json(path, record):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, allow_nan=False) + "\n")


class ActionHistogramMetric:
    def __init__(self, window=10, threshold=0.95):
        self.history: deque[torch.Tensor] = deque(maxlen=window)
        self.threshold = threshold

    def update(self, actions):
        histogram = torch.bincount(actions.flatten().cpu(), minlength=4).float()
        self.history.append(histogram)
        window = torch.stack(list(self.history)).sum(0)
        proportions = window / window.sum()
        collapsed = (
            len(self.history) == self.history.maxlen and proportions.max().item() > self.threshold
        )
        if collapsed:
            warnings.warn(f"Action collapse: {proportions.tolist()}", stacklevel=2)
        return (histogram / histogram.sum()).tolist(), collapsed

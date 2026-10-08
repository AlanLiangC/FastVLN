import time

import torch

from streamnav.contracts.action import NavigationAction
from streamnav.models.qwen35_kda.cache import state_bytes


@torch.no_grad()
def batch_steps(manager, requests, max_batch_size=8):
    if not requests or len(requests) > max_batch_size:
        raise ValueError("Invalid batch size")
    ids = [sid for sid, _ in requests]
    if len(set(ids)) != len(ids):
        raise ValueError("A session may occur only once in a batch")
    with manager.lock:
        manager._expire()
        sessions = [manager.sessions[sid] for sid in ids]
        with manager.autocast():
            logits, values, states = manager.policy.forward_batch(
                torch.stack([rgb for _, rgb in requests]), [s.state for s in sessions]
            )
        result = []
        for i, session in enumerate(sessions):
            session.state = states[i]
            session.last_access = time.monotonic()
            action = NavigationAction(logits[i].argmax(-1).item())
            result.append(
                {
                    "action": int(action),
                    "action_name": action.name,
                    "value": values[i].item(),
                    "probabilities": logits[i].softmax(-1).tolist(),
                    "step": states[i].step_index,
                    "state_bytes": state_bytes(states[i]),
                }
            )
        return result

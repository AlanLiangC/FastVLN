import time

import torch

from streamnav.contracts.action import NavigationAction
from streamnav.contracts.cache import state_bytes
from streamnav.contracts.object_hint import object_hint_token
from streamnav.evaluation.perception_records import perception_records


@torch.no_grad()
def batch_steps(manager, requests, max_batch_size=8):
    if not requests or len(requests) > max_batch_size:
        raise ValueError("Invalid batch size")
    ids = [request[0] for request in requests]
    if len(set(ids)) != len(ids):
        raise ValueError("A session may occur only once in a batch")
    with manager.lock:
        manager._expire()
        sessions = [manager.sessions[sid] for sid in ids]
        for request, session in zip(requests, sessions, strict=True):
            session.state = session.state.after_action(
                bool(request[3]) if len(request) > 3 else False,
                action=request[4] if len(request) > 4 else None,
            )
        hints = [
            object_hint_token(
                request[2] if len(request) > 2 else None,
                request[1].shape[1],
                request[1].shape[0],
                frame_id=session.state.step_index,
                episode_id=session.session_id,
            )
            for request, session in zip(requests, sessions, strict=True)
        ]
        pointing = getattr(manager.policy, "supports_object_hints", False)
        if any(hints) and not pointing:
            raise ValueError("Loaded checkpoint has no object-hint input")
        with manager.autocast():
            output = manager.policy.forward_batch(
                torch.stack([request[1] for request in requests]),
                [s.state for s in sessions],
                **({"hint_tokens": hints, "return_perception": True} if pointing else {}),
            )
            logits, values, states = output[:3]
        points = perception_records(output[3]) if pointing else []
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
                    **(
                        {"perception": points[i], "object_hint_token": hints[i]} if pointing else {}
                    ),
                }
            )
        return result

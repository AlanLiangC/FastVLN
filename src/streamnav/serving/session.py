import threading
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from streamnav.contracts.action import NavigationAction
from streamnav.contracts.cache import state_bytes
from streamnav.contracts.object_hint import object_hint_token
from streamnav.contracts.state import StreamingState
from streamnav.evaluation.perception_records import perception_records


@dataclass
class NavigationSession:
    session_id: str
    instruction: str
    state: StreamingState
    last_access: float


class SessionManager:
    def __init__(self, policy, max_sessions=16, ttl_s=900):
        self.policy = policy.eval()
        self.max_sessions, self.ttl_s = max_sessions, ttl_s
        self.sessions = {}
        self.lock = threading.RLock()

    def autocast(self):
        dtype = getattr(self.policy, "inference_compute_dtype", None)
        if dtype is None:
            return nullcontext()
        return torch.autocast(
            device_type=self.policy.backbone.device.type,
            dtype=dtype,
            enabled=dtype != torch.float32,
            # The viewer shares a GPU with a learner. Keeping a second, whole
            # BF16 weight copy until context exit adds ~1.6 GiB unnecessarily.
            cache_enabled=False,
        )

    def _expire(self):
        now = time.monotonic()
        for key in list(self.sessions):
            if now - self.sessions[key].last_access > self.ttl_s:
                del self.sessions[key]

    @torch.no_grad()
    def create(self, instruction):
        if not instruction.strip() or len(instruction) > 2048:
            raise ValueError("Instruction must contain 1–2048 characters")
        with self.lock:
            self._expire()
            if len(self.sessions) >= self.max_sessions:
                raise ValueError("Session capacity reached")
            sid = uuid.uuid4().hex
            with self.autocast():
                state = self.policy.start_episode(sid, instruction)
            self.sessions[sid] = NavigationSession(sid, instruction, state, time.monotonic())
            return sid

    @torch.no_grad()
    def step(self, session_id, rgb, object_hint=None):
        with self.lock:
            self._expire()
            session = self.sessions[session_id]
            hint = object_hint_token(
                object_hint,
                rgb.shape[1],
                rgb.shape[0],
                frame_id=session.state.step_index,
                episode_id=session_id,
            )
            pointing = getattr(self.policy, "supports_object_hints", False)
            with self.autocast():
                if pointing:
                    logits, values, states, predictions = self.policy.forward_batch(
                        rgb[None], [session.state], return_perception=True, hint_tokens=[hint]
                    )
                    action = NavigationAction(int(logits[0].argmax()))
                    probabilities, value, state = (
                        logits[0].softmax(-1).tolist(),
                        float(values[0]),
                        states[0],
                    )
                else:
                    if hint:
                        raise ValueError("Loaded checkpoint has no object-hint input")
                    action, output = self.policy.act(rgb, session.state, deterministic=True)
                    probabilities, value, state = (
                        output.logits.softmax(-1).tolist(),
                        output.value.item(),
                        output.state,
                    )
            session.state = state
            session.last_access = time.monotonic()
            return {
                "action": int(action),
                "action_name": action.name,
                "value": value,
                "probabilities": probabilities,
                "step": state.step_index,
                "state_bytes": state_bytes(state),
                **(
                    {"perception": perception_records(predictions)[0], "object_hint_token": hint}
                    if pointing
                    else {}
                ),
            }

    @torch.no_grad()
    def reset(self, session_id, instruction):
        if not instruction.strip() or len(instruction) > 2048:
            raise ValueError("Instruction must contain 1–2048 characters")
        with self.lock:
            self._expire()
            session = self.sessions[session_id]
            session.instruction = instruction
            with self.autocast():
                session.state = self.policy.start_episode(session_id, instruction)
            session.last_access = time.monotonic()

    def close(self, session_id):
        with self.lock:
            self.sessions.pop(session_id, None)

    def replace_policy(self, policy):
        with self.lock:
            self.sessions.clear()
            self.policy = policy.eval()

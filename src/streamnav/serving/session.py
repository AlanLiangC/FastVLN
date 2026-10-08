import threading
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from streamnav.contracts.state import StreamingState
from streamnav.models.qwen35_kda.cache import state_bytes


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
    def step(self, session_id, rgb):
        with self.lock:
            self._expire()
            session = self.sessions[session_id]
            with self.autocast():
                action, output = self.policy.act(rgb, session.state, deterministic=True)
            session.state = output.state
            session.last_access = time.monotonic()
            return {
                "action": int(action),
                "action_name": action.name,
                "value": output.value.item(),
                "probabilities": output.logits.softmax(-1).tolist(),
                "step": output.state.step_index,
                "state_bytes": state_bytes(output.state),
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
                session.state = self.policy.reset(session_id, instruction)
            session.last_access = time.monotonic()

    def close(self, session_id):
        with self.lock:
            self.sessions.pop(session_id, None)

    def replace_policy(self, policy):
        with self.lock:
            self.sessions.clear()
            self.policy = policy.eval()

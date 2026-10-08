import json
import os
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import zmq

from streamnav.contracts.action import NavigationAction
from streamnav.errors import EpisodeMismatchError, OracleUnavailableError, StreamNavError


class HabitatClient:
    def __init__(self, config, worker_id=0, root=None):
        self.root = Path(root or os.environ.get("STREAMNAV_ROOT", Path.cwd()))
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.linger = 0
        self.socket.rcvtimeo = int(config.get("timeout_s", 120) * 1000)
        self.socket.sndtimeo = int(config.get("timeout_s", 120) * 1000)
        # Abstract Linux IPC path avoids long workspace names and filesystem junk.
        self.endpoint = f"ipc://@streamnav-{os.getpid()}-{uuid.uuid4().hex[:10]}"
        log_dir = self.root / "runtime/logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log = open(log_dir / f"habitat-{os.getpid()}-{worker_id}.log", "w")
        self.process = subprocess.Popen(
            [
                str(self.root / config.get("python", "runtime/habitat-env/bin/python")),
                str(self.root / "services/habitat_server/server.py"),
                "--endpoint",
                self.endpoint,
                "--config",
                json.dumps(
                    {
                        **config,
                        "seed": config.get("seed", 2025)
                        + (worker_id if isinstance(worker_id, int) else 0),
                    }
                ),
            ],
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": str(self.root / "src")},
        )
        self.socket.connect(self.endpoint)
        self.episode_id = None
        try:
            self.request("PING")
        except Exception:
            self.close(force=True)
            raise

    def request(self, command, **kwargs):
        self.socket.send_json({"command": command, **kwargs})
        try:
            header, data = self.socket.recv_multipart()
        except zmq.Again as exc:
            raise StreamNavError(f"Habitat timeout for {command}; see {self.log.name}") from exc
        info = json.loads(header)
        if not info.pop("ok"):
            if info.get("error_type") == "OracleUnavailableError":
                raise OracleUnavailableError(info["error"])
            raise StreamNavError(f"Habitat {command}: {info['error']}")
        if "rgb_shape" in info:
            info["rgb"] = torch.from_numpy(
                np.frombuffer(data, dtype=np.uint8).reshape(info.pop("rgb_shape")).copy()
            )
        return info

    def reset(self, episode):
        info = self.request("RESET", episode=episode.to_dict())
        self.episode_id = episode.uid
        if info["episode_id"] != self.episode_id:
            raise EpisodeMismatchError("Simulator reset returned a different episode")
        return info

    def step(self, action: NavigationAction):
        if not isinstance(action, NavigationAction):
            raise TypeError("Use NavigationAction at the environment boundary")
        info = self.request("STEP", action=int(action))
        if info["episode_id"] != self.episode_id:
            raise EpisodeMismatchError("Simulator response belongs to another episode")
        return info

    def get_oracle_action(self):
        return NavigationAction(self.request("GET_ORACLE_ACTION")["action"])

    def close(self, force=False):
        if self.process.poll() is None:
            if not force:
                try:
                    self.request("CLOSE")
                except (StreamNavError, zmq.ZMQError):
                    pass
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.socket.close()
        self.context.term()
        self.log.close()


class VectorHabitatEnvs:
    """One simulator process and one socket per environment, parallel RPC fanout."""

    def __init__(self, config, num_envs):
        self.clients = []
        self.executor = ThreadPoolExecutor(max_workers=num_envs)
        try:
            for i in range(num_envs):
                self.clients.append(HabitatClient(config, i))
        except Exception:
            self.close()
            raise

    def reset(self, episodes):
        return list(
            self.executor.map(
                lambda pair: pair[0].reset(pair[1]), zip(self.clients, episodes, strict=True)
            )
        )

    def step(self, actions):
        return list(
            self.executor.map(
                lambda pair: pair[0].step(pair[1]), zip(self.clients, actions, strict=True)
            )
        )

    def reset_at(self, episodes):
        """Reset completed slots concurrently without touching live episodes."""
        return dict(
            self.executor.map(
                lambda item: (item[0], self.clients[item[0]].reset(item[1])), episodes.items()
            )
        )

    def get_oracle_actions(self):
        def oracle(client):
            try:
                return client.get_oracle_action()
            except OracleUnavailableError as exc:
                return exc

        return list(self.executor.map(oracle, self.clients))

    def close(self):
        for client in self.clients:
            client.close()
        self.executor.shutdown(wait=True)

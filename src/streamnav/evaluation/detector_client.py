"""One isolated detector per GPU, batched across the navigation environments."""

import json
import os
import subprocess
import uuid
from pathlib import Path

import numpy as np
import torch
import zmq

from streamnav.evaluation.detector_points import detector_point


class DetectorClient:
    def __init__(self, directory):
        root = Path(os.environ.get("STREAMNAV_ROOT", Path.cwd()))
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.linger = 0
        self.socket.rcvtimeo = self.socket.sndtimeo = 180_000
        endpoint = f"ipc://@streamnav-detector-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        device = torch.cuda.current_device()
        detector_gpu = visible.split(",")[device] if visible else str(device)
        self.log = (directory / f"detector-{os.getpid()}.log").open("a")
        self.process = subprocess.Popen(
            [
                str(root / "runtime/ovsegdt-env/bin/python"),
                str(root / "tools/object_detector_worker.py"),
                "--endpoint",
                endpoint,
            ],
            cwd=root,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": detector_gpu,
                "PYTHONPATH": str(root / "tools"),
                "YOLO_CONFIG_DIR": str(root / "runtime/cache/ovsegdt/ultralytics"),
                "MPLCONFIGDIR": str(root / "runtime/cache/ovsegdt/matplotlib"),
            },
        )
        self.socket.connect(endpoint)
        try:
            self.request("PING")
        except Exception:
            self.close()
            raise

    def request(self, command, images=None, **fields):
        array = (
            np.stack(
                [np.asarray(x.cpu() if hasattr(x, "cpu") else x, dtype=np.uint8) for x in images]
            )
            if images is not None
            else None
        )
        self.socket.send_multipart(
            [
                json.dumps(
                    {
                        "command": command,
                        **fields,
                        **({"shape": list(array.shape)} if array is not None else {}),
                    }
                ).encode(),
                array.tobytes() if array is not None else b"",
            ]
        )
        header, data = self.socket.recv_multipart()
        result = json.loads(header)
        if not result["ok"]:
            raise RuntimeError("YOLOE detector: " + result["error"])
        return np.frombuffer(data, dtype=np.uint8).reshape(result["shape"]) if data else result

    def predict(self, images, categories, split):
        if not images:
            return []
        masks = self.request("DETECT", images=images, categories=categories, split=split)
        return [detector_point(mask) for mask in masks]

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.socket.close()
        self.context.term()
        self.log.close()

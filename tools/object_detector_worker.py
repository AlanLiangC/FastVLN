"""Isolated, batched YOLOE worker. No navigation policy or privileged annotations."""

import argparse
import json
import traceback

import numpy as np
import torch
import zmq
from ovsegdt_runtime import ObjectMaskDetector, bootstrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    args = parser.parse_args()
    bootstrap()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(args.endpoint)
    detector, split = None, None
    try:
        while True:
            header, pixels = socket.recv_multipart()
            request = json.loads(header)
            mask = b""
            try:
                if request["command"] == "PING":
                    result = {"ready": True, "model": "yoloe-v8l-seg", "input_size": 224}
                elif request["command"] == "DETECT":
                    if split != request["split"]:
                        detector = None
                        torch.cuda.empty_cache()
                        split = request["split"]
                        detector = ObjectMaskDetector(split)
                    images = np.frombuffer(pixels, dtype=np.uint8).reshape(request["shape"])
                    output = detector.predict_batch(images, request["categories"])
                    output = output.to(torch.uint8).cpu().numpy()
                    result, mask = {"shape": list(output.shape)}, output.tobytes()
                elif request["command"] == "CLOSE":
                    result = {"closed": True}
                else:
                    raise ValueError("Unknown detector command")
                socket.send_multipart([json.dumps({"ok": True, **result}).encode(), mask])
            except Exception as error:
                traceback.print_exc()
                socket.send_multipart(
                    [json.dumps({"ok": False, "error": str(error)}).encode(), b""]
                )
            if request["command"] == "CLOSE":
                break
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()

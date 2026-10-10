from __future__ import annotations

import argparse
import json
import traceback

import zmq
from env_factory import HabitatObjectNavEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    if not (args.endpoint.startswith("ipc://") or args.endpoint.startswith("tcp://127.0.0.1:")):
        raise ValueError("Simulator RPC must bind to a local endpoint")
    env = HabitatObjectNavEnv(json.loads(args.config))
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.linger = 0
    socket.bind(args.endpoint)
    try:
        while True:
            request = socket.recv_json()
            command = request.get("command")
            rgb = None
            try:
                if command == "PING":
                    info = {"ready": True, "protocol": 1}
                elif command == "RESET":
                    rgb, info = env.reset(request["episode"])
                elif command == "STEP":
                    rgb, info = env.step(request["action"])
                elif command == "GET_ORACLE_ACTION":
                    info = {"action": int(env.oracle())}
                elif command == "GET_ORACLE_SUPERVISION":
                    info = env.oracle_supervision()
                elif command == "CLOSE":
                    info = {"closed": True}
                else:
                    raise ValueError(f"Unknown command {command}")
                if rgb is not None:
                    info["rgb_shape"] = list(rgb.shape)
                socket.send_multipart(
                    [
                        json.dumps({"ok": True, **info}).encode(),
                        rgb.tobytes() if rgb is not None else b"",
                    ]
                )
            except Exception as exc:
                traceback.print_exc()
                socket.send_multipart(
                    [
                        json.dumps(
                            {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
                        ).encode(),
                        b"",
                    ]
                )
            if command == "CLOSE":
                break
    finally:
        env.close()
        socket.close()
        context.term()


if __name__ == "__main__":
    main()

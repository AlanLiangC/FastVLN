"""Local OVSegDT inference worker under the isolated Python 3.9 environment."""

import argparse
import json
import pickle
import traceback

import numpy as np
import torch
import zmq
from ovsegdt_runtime import (
    ASSETS,
    ObjectMaskDetector,
    load_policy,
    prepare_observation,
    start_state,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    args = parser.parse_args()
    if not args.endpoint.startswith("ipc://@streamnav-"):
        raise ValueError("Only private local IPC endpoints are supported")
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    policy, _, report = load_policy(original_eval=True)
    distributions = {}

    def capture_distribution(module, inputs, output):
        distributions["current"] = output

    policy.action_distribution.register_forward_hook(capture_distribution)
    with (ASSETS / "siglip.pkl").open("rb") as handle:
        goals = pickle.load(handle)
    detector, split, category, cache, previous = None, None, None, None, None
    expected_frame = 0
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.linger = 0
    socket.bind(args.endpoint)
    try:
        while True:
            header, pixels = socket.recv_multipart()
            request = json.loads(header)
            command = request["command"]
            mask = None
            try:
                if command == "PING":
                    result = {"ready": True, **report}
                elif command == "RESET":
                    if split != request["split"]:
                        split = request["split"]
                        if detector is not None:
                            del detector
                            torch.cuda.empty_cache()
                        detector = ObjectMaskDetector(split)
                    category = request["category"]
                    if category not in goals:
                        raise ValueError("Missing goal category: " + category)
                    torch.manual_seed(request["seed"])
                    torch.cuda.manual_seed_all(request["seed"])
                    cache, previous = start_state(policy)
                    previous_mask = torch.zeros(1, 128, device="cuda")
                    expected_frame = 0
                    mapped_category = detector.categories[detector.target_class(category)]
                    result = {
                        "reset": True,
                        "category": category,
                        "vocabulary": split,
                        "detector_category": mapped_category,
                    }
                elif command in ("DETECT", "STEP"):
                    if detector is None:
                        raise ValueError("RESET required before inference")
                    rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(request["rgb_shape"]).copy()
                    mask = detector.predict(rgb, category)
                    result = {
                        "mask_shape": list(mask.shape),
                        "mask_pixels": int(mask.sum()),
                        "detector_category": mapped_category,
                    }
                    if command == "STEP":
                        if request["frame_id"] != expected_frame:
                            raise ValueError("Unexpected frame index for the current KV cache")
                        # Call the author's act(), carrying both previous action
                        # and mask state. Sampling remains enabled by default.
                        with torch.inference_mode():
                            observation = prepare_observation(
                                rgb, mask, goals[category], expected_frame
                            )
                            value, action, _, cache, previous_mask = policy.act(
                                observation,
                                cache,
                                previous,
                                previous_mask,
                                torch.tensor(
                                    [[expected_frame > 0]], device="cuda", dtype=torch.bool
                                ),
                                deterministic=request.get("deterministic", False),
                            )
                            probabilities = distributions["current"].probs
                        previous = action
                        expected_frame += 1
                        result.update(
                            action=int(action.item()),
                            action_probabilities=probabilities[0].cpu().tolist(),
                            value=float(value.item()),
                            cache_step=expected_frame,
                        )
                elif command == "CLOSE":
                    result = {"closed": True}
                else:
                    raise ValueError("Unknown command: " + command)
                socket.send_multipart(
                    [
                        json.dumps({"ok": True, **result}).encode(),
                        mask.to(torch.uint8).cpu().numpy().tobytes() if mask is not None else b"",
                    ]
                )
            except Exception as exc:
                traceback.print_exc()
                socket.send_multipart([json.dumps({"ok": False, "error": str(exc)}).encode(), b""])
            if command == "CLOSE":
                break
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()

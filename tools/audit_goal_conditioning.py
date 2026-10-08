"""Compare goals on identical real viewer frames; this is not a navigation score."""

import argparse
import json
import statistics
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 5, 15])
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    def call(route, payload=None, method=None):
        request = urllib.request.Request(
            args.url + route,
            json.dumps(payload).encode() if payload is not None else None,
            {"Content-Type": "application/json"},
            method=method,
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)

    reports = []
    for index in args.indices:
        observation = call("/demo/reset", {"episode_index": index, "split_index": args.split})
        first = {k: v for k, v in observation.items() if k != "rgb_base64"}
        goals = [observation["instruction"], "Find a toilet.", "Find a lamp."]
        sessions, records = [], []
        try:
            for goal in goals:
                sessions.append(call("/sessions/start", {"instruction": goal})["session_id"])
            for step in range(args.max_steps):
                outputs = call(
                    "/batch_step",
                    [
                        {"session_id": sid, "rgb_base64": observation["rgb_base64"]}
                        for sid in sessions
                    ],
                )
                probabilities = [o["probabilities"] for o in outputs]
                records.append(
                    {
                        "step": step + 1,
                        "probabilities": probabilities,
                        "actions": [o["action"] for o in outputs],
                        "total_variation": [
                            sum(abs(a - b) for a, b in zip(probabilities[0], other, strict=True))
                            / 2
                            for other in probabilities[1:]
                        ],
                    }
                )
                observation = call("/demo/step", {})
                if observation["done"]:
                    break
        finally:
            for sid in sessions:
                call(f"/sessions/{sid}", method="DELETE")
        late = records[10:]
        summary = {
            "index": index,
            "frames": len(records),
            "goals": goals,
            "first_frame_tv": records[0]["total_variation"],
            "late_mean_tv": statistics.mean(v for r in late for v in r["total_variation"])
            if late
            else None,
            "argmax_disagreement_fraction": statistics.mean(
                r["actions"][0] != a for r in records for a in r["actions"][1:]
            ),
        }
        reports.append({"summary": summary, "episode": first, "frames": records})
        print(json.dumps(summary), flush=True)
    Path(args.output).write_text(
        json.dumps(
            {
                "protocol": "Same JPEG frames from autonomous viewer trajectory for all goals; independent persistent sessions. Sensitivity diagnostic, not evidence of correct target grounding or SR.",
                "episodes": reports,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

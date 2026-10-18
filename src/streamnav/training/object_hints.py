"""Simulate optional detector points; absence is never supplied as truth input."""

import hashlib
import random

from streamnav.contracts.perception import GRID_HEIGHT, GRID_WIDTH


def simulated_object_hint_packet(
    label, rng, *, dropout=0.2, jitter=1, false_positive=0.05, constrain_to_support=False
):
    """Retain simulation provenance for diagnostics; the actor receives only token."""
    if rng.random() < dropout:
        return {"token": 0, "source": "dropout"}
    if rng.random() < false_positive:
        return {"token": rng.randint(1, GRID_WIDTH * GRID_HEIGHT), "source": "false_positive"}
    if not label or not label.get("opos_valid") or not label.get("opos", 0):
        return {"token": 0, "source": "unavailable"}
    y, x = divmod(int(label["opos"]) - 1, GRID_WIDTH)
    x = min(GRID_WIDTH - 1, max(0, x + rng.randint(-jitter, jitter)))
    y = min(GRID_HEIGHT - 1, max(0, y + rng.randint(-jitter, jitter)))
    token = 1 + y * GRID_WIDTH + x
    if constrain_to_support and token not in label.get("opos_targets", [label["opos"]]):
        token = int(label["opos"])
    return {"token": token, "source": "instance_jitter" if jitter else "instance_point"}


def simulated_object_hint(
    label, rng, *, dropout=0.2, jitter=1, false_positive=0.05, constrain_to_support=False
):
    return simulated_object_hint_packet(
        label,
        rng,
        dropout=dropout,
        jitter=jitter,
        false_positive=false_positive,
        constrain_to_support=constrain_to_support,
    )["token"]


def episode_hint_enabled(episode_id, seed, probability=0.5):
    value = int(hashlib.sha256(f"{seed}:{episode_id}".encode()).hexdigest()[:16], 16) / 2**64
    return value < probability


def evaluation_object_hint(label, episode_id, frame_id, seed, config):
    return evaluation_object_hint_packet(label, episode_id, frame_id, seed, config)["token"]


def evaluation_object_hint_packet(label, episode_id, frame_id, seed, config):
    key = f"{seed}:{episode_id}:{frame_id}"
    rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
    return simulated_object_hint_packet(label, rng, **config)

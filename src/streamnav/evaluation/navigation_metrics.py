def aggregate_metrics(episodes):
    keys: tuple[str, ...] = (
        "success",
        "spl",
        "soft_spl",
        "distance_to_goal",
        "episode_length",
        "collision_rate",
    )
    if not episodes:
        raise ValueError("Evaluation produced no episodes")
    extra = ("success_strict_0_1", "oracle_success", "min_distance_to_goal")
    keys = (*keys, *(k for k in extra if all(k in e for e in episodes)))
    return {key: sum(e[key] for e in episodes) / len(episodes) for key in keys}

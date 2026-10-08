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
    result = {key: sum(e[key] for e in episodes) / len(episodes) for key in keys}
    if all("near_goal_steps" in e for e in episodes):
        steps = sum(e["near_goal_steps"] for e in episodes)
        maxima = [
            e["near_goal_stop_probability_max"]
            for e in episodes
            if e["near_goal_stop_probability_max"] is not None
        ]
        result.update(
            near_goal_steps=steps,
            near_goal_stop_probability_mean=sum(
                e["near_goal_stop_probability_sum"] for e in episodes
            )
            / steps
            if steps
            else None,
            near_goal_stop_probability_max=max(maxima) if maxima else None,
            false_stop_rate=sum(e["false_stop"] for e in episodes) / len(episodes),
            goal_reached_without_stop_rate=sum(
                e["oracle_success"] > 0 and e["success"] == 0 for e in episodes
            )
            / len(episodes),
        )
    return result

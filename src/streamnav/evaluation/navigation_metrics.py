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
    if all("path_length" in e for e in episodes):
        result.update(
            path_length=sum(e["path_length"] for e in episodes) / len(episodes),
            stationary_episode_rate=sum(e["path_length"] < 0.01 for e in episodes) / len(episodes),
        )
    if all("max_consecutive_turns" in e for e in episodes):
        result["long_turn_episode_rate"] = sum(
            e["max_consecutive_turns"] >= 100 for e in episodes
        ) / len(episodes)
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
    if all("max_alternating_turns" in e for e in episodes):
        result.update(
            alternating_turn_episode_rate=sum(e["max_alternating_turns"] >= 8 for e in episodes)
            / len(episodes),
            long_alternating_turn_episode_rate=sum(
                e["max_alternating_turns"] >= 100 for e in episodes
            )
            / len(episodes),
        )
    if all("max_forward_collision_streak" in e for e in episodes):
        after_collision = sum(e["after_collision_steps"] for e in episodes)
        result.update(
            blocked_forward_episode_rate=sum(
                e["max_forward_collision_streak"] >= 8 for e in episodes
            )
            / len(episodes),
            long_blocked_forward_episode_rate=sum(
                e["max_forward_collision_streak"] >= 100 for e in episodes
            )
            / len(episodes),
            max_forward_collision_streak=max(e["max_forward_collision_streak"] for e in episodes),
            after_collision_forward_rate=sum(e["after_collision_forward_steps"] for e in episodes)
            / after_collision
            if after_collision
            else None,
        )
    return result

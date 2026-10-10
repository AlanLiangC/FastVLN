"""Training-only expert/greedy slots; their actions never enter the PPO objective."""

from streamnav.contracts.action import NavigationAction


def auxiliary_il_config(config=None):
    return {
        "num_envs": 0,
        "expert_episode_period": 2,
        "alternating_turn_limit": 8,
        "consecutive_turn_limit": 16,
        "recovery_steps": 32,
        **(config or {}),
    }


class AuxiliaryILController:
    def __init__(self, config, total_envs):
        self.config = auxiliary_il_config(config)
        self.num_envs = self.config["num_envs"]
        if not isinstance(self.num_envs, int) or not 0 <= self.num_envs < total_envs:
            raise ValueError("Auxiliary IL must leave at least one sampled PPO environment")
        for key in (
            "expert_episode_period",
            "alternating_turn_limit",
            "consecutive_turn_limit",
            "recovery_steps",
        ):
            if not isinstance(self.config[key], int) or self.config[key] < 1:
                raise ValueError(f"Auxiliary IL {key} must be a positive integer")
        self.episode_counts = [-1] * self.num_envs
        self.recovery_remaining = [0] * self.num_envs
        self.last_action = [None] * self.num_envs
        self.alternating_turns = [0] * self.num_envs
        self.consecutive_turns = [0] * self.num_envs

    def reset(self, env):
        if env >= self.num_envs:
            return
        self.episode_counts[env] += 1
        self.recovery_remaining[env] = 0
        self._clear_turns(env)

    def _clear_turns(self, env):
        self.last_action[env] = None
        self.alternating_turns[env] = self.consecutive_turns[env] = 0

    def select(self, env, greedy, expert):
        if env >= self.num_envs:
            raise ValueError("Only auxiliary slots may use expert/greedy control")
        expert_episode = self.episode_counts[env] % self.config["expert_episode_period"] == 0
        use_expert = expert_episode or self.recovery_remaining[env] > 0
        if self.recovery_remaining[env] > 0:
            self.recovery_remaining[env] -= 1
        return (expert if use_expert else greedy), use_expert

    def observe(self, env, executed, used_expert):
        if env >= self.num_envs:
            return False
        if used_expert:
            self._clear_turns(env)
            return False
        turns = (NavigationAction.TURN_LEFT, NavigationAction.TURN_RIGHT)
        if executed in turns:
            previous = self.last_action[env]
            self.consecutive_turns[env] += 1
            self.alternating_turns[env] = (
                self.alternating_turns[env] + 1 if previous in turns and previous != executed else 1
            )
            self.last_action[env] = executed
        else:
            self._clear_turns(env)
        if (
            self.alternating_turns[env] >= self.config["alternating_turn_limit"]
            or self.consecutive_turns[env] >= self.config["consecutive_turn_limit"]
        ):
            self.recovery_remaining[env] = self.config["recovery_steps"]
            self._clear_turns(env)
            return True
        return False

    def state_dict(self):
        # Simulator episodes/caches restart on resume. Only the episode schedule
        # continues; a fresh episode must not inherit a previous loop/recovery.
        return {"episode_counts": list(self.episode_counts)}

    def load_state_dict(self, state):
        counts = state["episode_counts"]
        if len(counts) != self.num_envs or any(not isinstance(x, int) for x in counts):
            raise ValueError("Auxiliary IL checkpoint slot count differs")
        self.episode_counts = list(counts)

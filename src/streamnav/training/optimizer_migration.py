"""Retain existing Adam states when an explicit fork adds planning or hint parameters."""


def load_optimizer_with_added_groups(optimizer, saved, allowed_new=("planning", "hint", "arrival")):
    current = optimizer.state_dict()
    old_groups = {g["name"]: g for g in saved["param_groups"]}
    if not set(old_groups).issubset({g["name"] for g in current["param_groups"]}):
        raise ValueError("Optimizer fork cannot remove parameter groups")
    state, groups = {}, []
    for group in current["param_groups"]:
        old = old_groups.get(group["name"])
        if old is None:
            if group["name"] not in allowed_new:
                raise ValueError("Unexpected new optimizer group")
            groups.append(group)
            continue
        if len(old["params"]) != len(group["params"]):
            raise ValueError("Existing optimizer parameter group changed shape")
        for before, after in zip(old["params"], group["params"], strict=True):
            if before in saved["state"]:
                state[after] = saved["state"][before]
        groups.append({**old, "params": group["params"]})
    optimizer.load_state_dict({"state": state, "param_groups": groups})

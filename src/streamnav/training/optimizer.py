import math

import torch


def optimizer_epsilon(config, role):
    value = config.get("backbone_optimizer_eps") if role == "backbone" else None
    value = config.get("optimizer_eps", 1e-5) if value is None else value
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Adam epsilon must be finite and positive")
    return value


def build_optimizer(policy, config):
    vision = [p for p in policy.backbone.vision.parameters() if p.requires_grad]
    vision_ids = {id(p) for p in vision}
    backbone = [
        p for p in policy.backbone.parameters() if p.requires_grad and id(p) not in vision_ids
    ]
    # The selected upstream finetune configuration has critic-only update 1,
    # then actor/state encoder LR 2.5e-4 from update 2 onward. No weight decay.
    initial_freeze = config.get("actor_warmup_updates", 0)
    if initial_freeze:
        for p in backbone + list(policy.actor_critic.actor.parameters()):
            p.requires_grad_(False)
    optimizer_type = (
        torch.optim.Adam if config.get("optimizer", "adam") == "adam" else torch.optim.AdamW
    )
    groups = [
        {
            "params": policy.actor_critic.critic.parameters(),
            "lr": config.get("critic_lr", config["head_lr"]),
            "role": "critic",
        },
        {
            "params": backbone,
            "lr": 0.0 if initial_freeze else config["backbone_lr"],
            "role": "backbone",
            "eps": optimizer_epsilon(config, "backbone"),
        },
        {
            "params": policy.actor_critic.actor.parameters(),
            "lr": 0.0 if initial_freeze else config["head_lr"],
            "role": "actor",
        },
        {
            "params": vision,
            "lr": config.get("vision_lr", config["backbone_lr"]),
            "role": "vision",
        },
    ]
    if getattr(policy, "perception", None) is not None:
        groups.append(
            {
                "params": policy.perception.parameters(),
                "lr": config.get("perception_lr", config["head_lr"]),
                "role": "perception",
            }
        )
    return optimizer_type(
        groups,
        weight_decay=config["weight_decay"],
        eps=optimizer_epsilon(config, "actor"),
        fused=config.get("fused_optimizer", True) and backbone[0].is_cuda,
    )


class OVSegDTLRScheduler:
    """PIRLNav schedule for start_warmup=start_update=1 in the reference YAML."""

    def __init__(self, optimizer, config):
        self.optimizer, self.config, self.update = optimizer, config, 0

    def step(self):
        self.update += 1
        self._apply()

    def _apply(self):
        if self.update < self.config.get("actor_warmup_updates", 0):
            return
        for group in self.optimizer.param_groups:
            if group["role"] in ("backbone", "actor"):
                for parameter in group["params"]:
                    parameter.requires_grad_(True)
                group["lr"] = (
                    self.config["backbone_lr"]
                    if group["role"] == "backbone"
                    else self.config["head_lr"]
                )

    def state_dict(self):
        return {"update": self.update}

    def load_state_dict(self, state):
        self.update = state["update"]
        self._apply()

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


def gradient_norm(parameters):
    gradients = [p.grad.detach().float() for p in parameters if p.grad is not None]
    # One foreach launch avoids hundreds of tiny norm/square kernels per branch.
    norms = torch._foreach_norm(gradients) if gradients else []
    return torch.linalg.vector_norm(torch.stack(list(norms))).item() if norms else 0.0

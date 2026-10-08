import torch


def build_optimizer(policy, config):
    vision = [p for p in policy.backbone.vision.parameters() if p.requires_grad]
    vision_ids = {id(p) for p in vision}
    backbone = [
        p for p in policy.backbone.parameters() if p.requires_grad and id(p) not in vision_ids
    ]
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": config["backbone_lr"]},
            {"params": vision, "lr": config.get("vision_lr", config["backbone_lr"])},
            {"params": policy.actor_critic.parameters(), "lr": config["head_lr"]},
        ],
        weight_decay=config["weight_decay"],
        fused=config.get("fused_optimizer", True) and backbone[0].is_cuda,
    )


def gradient_norm(parameters):
    gradients = [p.grad.detach().float() for p in parameters if p.grad is not None]
    # One foreach launch avoids hundreds of tiny norm/square kernels per branch.
    norms = torch._foreach_norm(gradients) if gradients else []
    return torch.linalg.vector_norm(torch.stack(list(norms))).item() if norms else 0.0

"""Matched native-Qwen vs converted-KDA single-frame capacity/view controls.

Both start from the same pretrained Qwen weights, NAV and actor initialization.
No navigation training checkpoint, semantic input or heldout optimizer data.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from audit_architecture_capacity import configuration, set_memory_timescales, write_json
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3_5ForConditionalGeneration

from streamnav.training.checkpoint import load_policy
from streamnav.utils.seed import seed_everything


def metrics(logits, labels):
    predicted = logits.argmax(-1)
    return {
        "cross_entropy": F.cross_entropy(logits, labels).item(),
        "accuracy": (predicted == labels).float().mean().item(),
        "stop_recall": (predicted[labels == 0] == 0).float().mean().item(),
        "nonstop_recall": (predicted[labels != 0] != 0).float().mean().item(),
        "stop_binary_accuracy": ((predicted == 0) == (labels == 0)).float().mean().item(),
        "confusion": torch.bincount(labels * 6 + predicted, minlength=36).view(6, 6).cpu().tolist(),
    }


class FrameReadout(nn.Module):
    def __init__(self, args):
        super().__init__()
        config = configuration(args.checkpoint, args.device)
        config.update(checkpoint="checkpoints/qwen35_0p8b_kda", device="cpu")
        base = load_policy(config, training=True)
        self.tokenizer = base.backbone.tokenizer
        self.config = base.backbone.config
        self.nav = nn.Parameter(base.backbone.nav_token.detach().clone())
        self.actor = copy.deepcopy(base.actor_critic.actor)
        self.variant = args.variant
        self.disable_language_rope = getattr(args, "disable_language_rope", False)
        self.natural_readout = getattr(args, "natural_readout", False)
        if args.variant == "native":
            source = json.loads(Path("checkpoints/qwen35_0p8b_kda/kda_layout.json").read_text())[
                "source"
            ]
            original = Qwen3_5ForConditionalGeneration.from_pretrained(
                source, dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
            )
            self.body = original.model
            self.body.visual.requires_grad_(False)
            # Same FLA GDN kernel as the streaming student; the native full
            # attention and multimodal RoPE remain untouched. These are whole
            # zero-state sequences, so no native multi-token cache issue.
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule

            for layer in self.body.language_model.layers:
                if layer.layer_type == "linear_attention":
                    layer.linear_attn.chunk_gated_delta_rule = chunk_gated_delta_rule
            self.body.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            self.body = base.backbone
            if args.variant == "kda_long_memory":
                set_memory_timescales(self.body)
        self.to(args.device)
        self.prefix_ids = {}

    @property
    def embedding(self):
        return (
            self.body.language_model.embed_tokens
            if self.variant == "native"
            else self.body.embeddings
        )

    def forward(self, clips, *, return_hidden=False):
        tokens, dummy_ids = [], []
        device = self.nav.device
        for clip in clips:
            goal = clip["goal"]
            if self.natural_readout:
                if goal not in self.prefix_ids:
                    prompt = self.tokenizer.apply_chat_template(
                        [
                            {
                                "role": "system",
                                "content": "You are a fast object navigation policy. Navigate to the requested object.",
                            },
                            {
                                "role": "user",
                                "content": [{"type": "image"}, {"type": "text", "text": goal}],
                            },
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    ids = torch.tensor(self.tokenizer(prompt).input_ids, device=device)
                    positions = (ids == self.config.image_token_id).nonzero().flatten()
                    if positions.numel() != 1:
                        raise ValueError("Natural prompt must have exactly one image placeholder")
                    position = positions.item()
                    self.prefix_ids[goal] = ids[:position], ids[position + 1 :]
                before, after = self.prefix_ids[goal]
                visual = clip["visual"][0].to(device)
                tokens.append(torch.cat((self.embedding(before), visual, self.embedding(after))))
                dummy_ids.append(
                    torch.cat(
                        (
                            before,
                            before.new_full((visual.shape[0],), self.config.image_token_id),
                            after,
                        )
                    )
                )
                continue
            if goal not in self.prefix_ids:
                prompt = self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": "You are a fast object navigation policy. Navigate to the requested object.",
                        },
                        {"role": "user", "content": goal},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                self.prefix_ids[goal] = (
                    torch.tensor(self.tokenizer(prompt).input_ids, device=device),
                    torch.tensor(
                        self.tokenizer(goal, add_special_tokens=False).input_ids, device=device
                    ),
                )
            prefix, goal_ids = self.prefix_ids[goal]
            marker_ids = torch.tensor(
                [self.config.vision_start_token_id, self.config.vision_end_token_id], device=device
            )
            markers = self.embedding(marker_ids)
            visual = clip["visual"][0].to(device)
            query = self.nav[0] + self.embedding(goal_ids).mean(0, keepdim=True)
            tokens.append(
                torch.cat((self.embedding(prefix), markers[:1], visual, markers[1:], query))
            )
            image_ids = torch.tensor(
                [self.config.vision_start_token_id]
                + [self.config.image_token_id] * visual.shape[0]
                + [self.config.vision_end_token_id, self.tokenizer.eos_token_id],
                device=device,
            )
            dummy_ids.append(torch.cat((prefix, image_ids)))
        # The streaming kernels have no padding mask. Zero embeddings can
        # amplify backward roundoff through repeated normalizations, even when
        # the loss reads an earlier NAV token. Production never pads its
        # episode prefills or 138-token frames. Match both controls by length.
        groups = {}
        for index, sequence in enumerate(tokens):
            groups.setdefault(sequence.shape[0], []).append(index)
        readouts = [None] * len(clips)
        for indices in groups.values():
            batch = torch.stack([tokens[i] for i in indices])
            if self.variant == "native":
                ids = torch.stack([dummy_ids[i] for i in indices])
                mask = torch.ones_like(ids)
                positions, _ = self.body.get_rope_index(
                    ids,
                    mm_token_type_ids=(ids == self.config.image_token_id).int(),
                    image_grid_thw=torch.tensor([[1, 18, 30]], device=device).expand(
                        len(indices), -1
                    ),
                    attention_mask=mask,
                )
                if self.disable_language_rope:
                    positions = torch.zeros_like(positions)
                hidden = self.body.language_model(
                    inputs_embeds=batch,
                    position_ids=positions,
                    attention_mask=mask,
                    use_cache=False,
                ).last_hidden_state
            else:
                hidden, _ = self.body.recurrent_forward(batch, mode="chunk")
            for index, readout in zip(indices, hidden[:, -1], strict=True):
                readouts[index] = readout
        readout = torch.stack(readouts)
        return readout if return_hidden else self.actor(readout).float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--variant", choices=["native", "kda", "kda_long_memory"], required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4971)
    parser.add_argument("--fixed-steps", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)
    torch.cuda.set_device(args.device)
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    clips, held = data["clips"], data["holdout_clips"]
    labels = torch.cat([c["actions"] for c in clips]).to(args.device)
    held_labels = torch.cat([c["actions"] for c in held]).to(args.device)
    model = FrameReadout(args)
    shared_init = hashlib.sha256()
    for value in (model.nav, *model.actor.state_dict().values()):
        shared_init.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    parameters = [p for p in model.body.parameters() if p.requires_grad] + [model.nav]
    optimizer = torch.optim.Adam(
        [{"params": parameters, "lr": 1e-5}, {"params": model.actor.parameters(), "lr": 2.5e-4}],
        eps=1e-5,
        fused=True,
    )
    history = []

    @torch.no_grad()
    def evaluate(clips, labels):
        model.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(clips)
        return metrics(logits, labels), logits

    initial, _ = evaluate(clips, labels)
    history.append({"step": 0, **initial})
    print(json.dumps(history[-1]), flush=True)
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(clips), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            parameters + list(model.actor.parameters()), 0.2, error_if_nonfinite=True
        )
        optimizer.step()
        if step % 10 == 0:
            row, _ = evaluate(clips, labels)
            history.append({"step": step, **row})
            print(json.dumps(history[-1]), flush=True)
            if not args.fixed_steps and row["accuracy"] == 1.0 and row["cross_entropy"] < 0.05:
                break
    final, train_logits = evaluate(clips, labels)
    holdout, held_logits = evaluate(held, held_labels)
    torch.save(
        {
            "train_logits": train_logits.cpu(),
            "train_labels": labels.cpu(),
            "holdout_logits": held_logits.cpu(),
            "holdout_labels": held_labels.cpu(),
        },
        Path(args.output).with_suffix(".outputs.pt"),
    )
    write_json(
        args.output,
        {
            "protocol": "Matched pretrained initialization, same NAV/actor, RGB/text goal-pair data and active learning rates. Native retains six full-attention layers with official MRoPE; native GDN uses student FLA kernel. Both process whole prompt+frame sequences grouped by exact length without padding and read final NAV. Pure IL; heldout yaw views never enter optimization. Does not test long-horizon streaming or unseen scenes/categories.",
            "variant": args.variant,
            "seed": args.seed,
            "fixed_step_budget": args.steps if args.fixed_steps else None,
            "shared_nav_actor_initialization_sha256": shared_init.hexdigest(),
            "history": history,
            "final": final,
            "heldout_view_perturbations": holdout,
        },
    )


if __name__ == "__main__":
    main()

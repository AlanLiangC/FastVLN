"""Native Qwen control for repeated stationary image/goal chat turns.

Evaluate full conversations to avoid native multi-token cache limitations.
Same saved single-frame diagnostic head, no optimizer or semantic inputs.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from audit_architecture_capacity import write_json
from audit_cross_scene_capacity import dataset, score
from audit_native_goal_pair_fit import FrameReadout


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:6")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_device(args.device)
    torch.manual_seed(10913)
    model = FrameReadout(
        SimpleNamespace(
            checkpoint=args.checkpoint,
            variant="native",
            device=args.device,
            natural_readout=True,
        )
    ).eval()
    root = Path(args.data_root)
    reference = torch.load(
        root / "fit_natural_native.features.pt", map_location="cpu", weights_only=False
    )
    model.actor.load_state_dict(reference["probe"])
    data = dataset(root)
    tokenizer = model.tokenizer
    close = torch.tensor(
        tokenizer("<|im_end|>\n", add_special_tokens=False).input_ids, device=args.device
    )
    parts = {}
    records, saved_features = {}, {}
    for split in ("train", "val_seen", "val_unseen"):
        extracted = {i: [] for i in (1, 4, 16)}
        for index, clip in enumerate(data[split]):
            goal = clip["goal"]
            if goal not in parts:
                prompt = tokenizer.apply_chat_template(
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
                ids = torch.tensor(
                    tokenizer(prompt, add_special_tokens=False).input_ids, device=args.device
                )
                image = (ids == model.config.image_token_id).nonzero().flatten().item()
                user = (
                    (ids[:image] == tokenizer.convert_tokens_to_ids("<|im_start|>"))
                    .nonzero()
                    .flatten()[1]
                    .item()
                )
                parts[goal] = ids[:image], ids[image + 1 :], ids[user:image]
            before, after, user_prefix = parts[goal]
            visual = clip["visual"][0].to(args.device)
            image_ids = before.new_full((visual.shape[0],), model.config.image_token_id)
            tokens = torch.cat((model.embedding(before), visual, model.embedding(after)))
            ids = torch.cat((before, image_ids, after))
            continuation_ids = torch.cat((close, user_prefix, image_ids, after))
            continuation = torch.cat(
                (model.embedding(torch.cat((close, user_prefix))), visual, model.embedding(after))
            )
            for frames in extracted:
                inputs = torch.cat([tokens] + [continuation] * (frames - 1)).unsqueeze(0)
                input_ids = torch.cat([ids] + [continuation_ids] * (frames - 1)).unsqueeze(0)
                mask = torch.ones_like(input_ids)
                positions, _ = model.body.get_rope_index(
                    input_ids,
                    mm_token_type_ids=(input_ids == model.config.image_token_id).int(),
                    image_grid_thw=torch.tensor([[1, 18, 30]], device=args.device).expand(
                        frames, -1
                    ),
                    attention_mask=mask,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = model.body.language_model(
                        inputs_embeds=inputs,
                        position_ids=positions,
                        attention_mask=mask,
                        use_cache=False,
                    ).last_hidden_state[:, -1]
                extracted[frames].append(hidden.float().cpu())
            if index % 24 == 23:
                print(json.dumps({"split": split, "frames_processed": index + 1}), flush=True)
        saved_features[split] = {i: torch.cat(rows) for i, rows in extracted.items()}
        records[split] = {
            i: score(
                torch.nn.functional.linear(x, model.actor.weight.cpu(), model.actor.bias.cpu()),
                data[split],
            )
            for i, x in saved_features[split].items()
        }
        print(json.dumps({split: records[split]}), flush=True)
    torch.save(saved_features, Path(args.output).with_suffix(".features.pt"))
    write_json(args.output, {"protocol": __doc__, "records": records})


if __name__ == "__main__":
    main()

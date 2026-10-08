import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from streamnav.models.qwen35_kda.backbone import Qwen35KDABackbone


def convert(source, output, seed=17):
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    torch.manual_seed(seed)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty conversion: {output}")
    output.mkdir(parents=True, exist_ok=True)
    original = Qwen3_5ForConditionalGeneration.from_pretrained(
        source, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    tokenizer = AutoTokenizer.from_pretrained(source)
    backbone = Qwen35KDABackbone(original.model, tokenizer).to(dtype=torch.bfloat16)
    replaced = [
        i for i, t in enumerate(original.config.text_config.layer_types) if t == "full_attention"
    ]
    if replaced != [3, 7, 11, 15, 19, 23] or original.config.text_config.hidden_size != 1024:
        raise ValueError("Only Qwen3.5-0.8B is supported")
    layout = {
        "format_version": 1,
        "source": str(source),
        "source_config_sha256": hashlib.sha256(
            original.config.to_json_string().encode()
        ).hexdigest(),
        "replaced_full_attention_layers": replaced,
        "preserved_gdn_layers": 18,
        "recurrence": "KDA channel-wise decay + delta update",
        "seed": seed,
        "initialization": "Q/K/V/O and output gate copied; decay/beta new; no distillation",
        "rope": "removed from converted full-attention layers",
        "kernel": "fla",
        "versions": {
            n: importlib.metadata.version(n)
            for n in ("transformers", "torch", "flash-linear-attention", "fla-core")
        },
    }
    revision_file = Path(source) / ".cache/huggingface/download/config.json.metadata"
    layout["source_revision"] = (
        revision_file.read_text().splitlines()[0]
        if revision_file.exists()
        else getattr(original.config, "_commit_hash", None)
    )
    original.config.save_pretrained(output)
    tokenizer.save_pretrained(output / "tokenizer")
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in backbone.state_dict().items()},
        str(output / "model.safetensors"),
    )
    (output / "kda_layout.json").write_text(json.dumps(layout, indent=2))
    return layout

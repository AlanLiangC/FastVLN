"""Build the original category-only SigLIP goal cache in the isolated OVSegDT environment."""

import json
import pickle
from pathlib import Path

import open_clip
import torch
from open_clip.tokenizer import HFTokenizer


def main():
    root = Path(__file__).resolve().parents[1]
    assets = root / "checkpoints/ovsegdt"
    categories = set()
    for path in (root / "third_party/OVSegDT").glob("cats_*.json"):
        categories.update(json.loads(path.read_text())["cats_text"])
    for path in (root / "runtime/data/manifests").glob("hm3d_ovon*.json"):
        categories.update(json.loads(path.read_text())["categories"])
    categories = sorted(categories)
    model = open_clip.create_model(
        "ViT-B-16-SigLIP-256",
        pretrained=str(assets / "siglip_text/open_clip_model.safetensors"),
        device="cuda",
    ).eval()
    tokenizer = HFTokenizer(str(assets / "siglip_text"), context_length=64, clean="canonicalize")
    embeddings = {}
    with torch.inference_mode():
        for start in range(0, len(categories), 32):
            batch = categories[start : start + 32]
            # Match ovon/utils/cache_clip_embeddings.py: raw category, no
            # navigation prompt and no extra output-vector normalization.
            output = model.encode_text(tokenizer(batch).cuda()).float().cpu().numpy()
            assert len(batch) == len(output)
            embeddings.update(zip(batch, output))
    path = assets / "siglip.pkl"
    with path.open("wb") as handle:
        pickle.dump(embeddings, handle)
    (assets / "categories.json").write_text(json.dumps(categories, indent=2) + "\n")
    print(json.dumps({"path": str(path), "categories": len(categories), "dimension": 768}))


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()

import argparse
import json
import time

import torch

from streamnav.models.policy.streaming_policy import StreamingObjectNavPolicy
from streamnav.models.qwen35_kda.backbone import Qwen35KDABackbone
from streamnav.models.qwen35_kda.cache import state_bytes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/qwen35_0p8b_kda")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.manual_seed(17)
    b = Qwen35KDABackbone.from_converted(args.checkpoint, device=args.device, image_size=128)
    p = StreamingObjectNavPolicy(b)
    t = time.monotonic()
    with torch.no_grad():
        s = p.start_episode("probe", "Find a chair.")
        print("prefill", time.monotonic() - t, flush=True)
        out = p.forward_step(torch.zeros(128, 128, 3, dtype=torch.uint8), s)
        print("forward", out.logits, out.value, flush=True)
    out = p.forward_step(torch.zeros(128, 128, 3, dtype=torch.uint8), out.state)
    loss = out.logits.square().mean() + out.value.square().mean()
    loss.backward()
    result = {
        "loss": loss.item(),
        "state_bytes": state_bytes(out.state),
        "elapsed_s": time.monotonic() - t,
        "kda_grad": b.layers[3].mixer.beta_proj.weight.grad.norm().item(),
        "vision_grad": b.vision.patch_embed.proj.weight.grad.norm().item(),
    }
    print(json.dumps(result), flush=True)

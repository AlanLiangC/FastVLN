import torch
from torch.nn import functional as F


def patchify(rgb: torch.Tensor, size: int | tuple[int, int] | list[int] = 224):
    """Qwen3.5 patches: RGB uint8 BHWC -> merge-ordered 2x16x16 patches.

    An integer retains legacy square resizing. A (height, width) pair preserves
    aspect ratio with letterboxing, then pads to the model's 32-pixel alignment.
    Native 270x480 observations are not resized: only 9 rows per side are added.
    """
    if isinstance(size, int):
        if size % 32 or size < 32:
            raise ValueError("Integer image_size must be a positive multiple of 32")
        target_h = target_w = size
    else:
        if len(size) != 2 or any(not isinstance(v, int) or v < 1 for v in size):
            raise ValueError("image_size must contain positive integer height and width")
        target_h, target_w = size
    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != torch.uint8:
        raise ValueError("Expected uint8 RGB [B,H,W,3]")
    x = rgb.permute(0, 3, 1, 2).float()
    if isinstance(size, int):
        resized = (target_h, target_w)
    else:
        scale = min(target_h / x.shape[-2], target_w / x.shape[-1])
        resized = (max(1, round(x.shape[-2] * scale)), max(1, round(x.shape[-1] * scale)))
    if x.shape[-2:] != resized:
        x = F.interpolate(x, resized, mode="bicubic", align_corners=False, antialias=True)
        x = x.clamp(0, 255)
    x = x / 127.5 - 1.0
    height, width = ((target_h + 31) // 32) * 32, ((target_w + 31) // 32) * 32
    pad_h, pad_w = height - x.shape[-2], width - x.shape[-1]
    x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
    b, c, _, _ = x.shape
    gh, gw = height // 16, width // 16
    x = x[:, :, None].expand(-1, -1, 2, -1, -1)
    x = x.reshape(b, c, 2, gh // 2, 2, 16, gw // 2, 2, 16)
    x = x.permute(0, 3, 6, 4, 7, 1, 2, 5, 8).reshape(b * gh * gw, c * 2 * 16 * 16)
    grid = torch.tensor([[1, gh, gw]], device=rgb.device).expand(b, -1)
    return x, grid

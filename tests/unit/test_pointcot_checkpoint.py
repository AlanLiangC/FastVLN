import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from streamnav.training.pointcot_trainer import load_adapters


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.row = nn.Parameter(torch.zeros(3))
        self.device = torch.device("cpu")

    def parameter_summary(self):
        return {"source_revision": {"revision": "pinned"}}


def test_checkpoint_checks_integrity_and_base(tmp_path):
    expected = torch.tensor([1.0, 2.0, 3.0])
    digest = hashlib.sha256(b"row" + expected.numpy().tobytes()).hexdigest()
    save_file({"row": expected}, str(tmp_path / "adapters.safetensors"))
    manifest = {
        "format": "qwen35-pointcot-adapters-v1",
        "base": {"revision": "pinned"},
        "adapter_sha256": digest,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    policy = Policy()
    load_adapters(policy, tmp_path)
    torch.testing.assert_close(policy.row, expected)
    save_file({"row": expected + 1}, str(tmp_path / "adapters.safetensors"))
    with pytest.raises(ValueError, match="checksum"):
        load_adapters(policy, tmp_path)
    manifest["base"] = {"revision": "different"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="base"):
        load_adapters(policy, tmp_path)

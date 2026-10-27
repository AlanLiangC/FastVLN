"""Load the author's model for inference without importing training-only launchers."""

from __future__ import annotations

import json
import os
import pickle
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party/OVSegDT"
ASSETS = ROOT / "checkpoints/ovsegdt"


class MetadataUnpickler(pickle.Unpickler):
    """Decode saved OmegaConf class metadata without registering trainer plugins.

    These types describe only the config schema. Model classes and tensor
    reconstruction still come from the original repository and PyTorch.
    """

    carriers = {}

    def find_class(self, module, name):
        if module.startswith(("ovon.", "habitat.", "habitat_baselines.")):
            key = (module, name)
            if key not in self.carriers:
                self.carriers[key] = type(name, (), {"__module__": module})
            return self.carriers[key]
        return super().find_class(module, name)


def read_checkpoint(path=ASSETS / "ckpt.5.pth"):
    reader = types.ModuleType("ovsegdt_metadata_pickle")
    reader.Unpickler = MetadataUnpickler
    reader.load = pickle.load
    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=reader)


def bootstrap():
    for path in (
        VENDOR,
        VENDOR / "habitat-lab/habitat-lab",
        VENDOR / "habitat-lab/habitat-baselines",
    ):
        sys.path.insert(0, str(path))
    # Their __init__ files eagerly import every training algorithm. Keep package
    # resolution pointed at the original sources, but load only required modules.
    for name, path in (
        ("ovon", VENDOR / "ovon"),
        ("habitat_baselines", VENDOR / "habitat-lab/habitat-baselines/habitat_baselines"),
    ):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            module.__file__ = str(path / "__init__.py")
            sys.modules[name] = module


@contextmanager
def vendor_directory(directory=VENDOR):
    previous = Path.cwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(previous)


def load_policy(checkpoint=ASSETS / "ckpt.5.pth", *, original_eval=False):
    bootstrap()
    import timm
    from gym import spaces
    from omegaconf import OmegaConf
    from ovon.models.transformer_policy import OVONTransformerPolicy

    saved = read_checkpoint(checkpoint)
    config = OmegaConf.to_container(saved["config"], resolve=False)
    settings = config["habitat_baselines"]["rl"]["policy"]
    if original_eval:
        import yaml

        evaluation = yaml.safe_load(
            (VENDOR / "config/experiments/transformer_rl_segm_loss-validation.yaml").read_text()
        )
        settings = OmegaConf.to_container(
            OmegaConf.merge(
                OmegaConf.create(settings), evaluation["habitat_baselines"]["rl"]["policy"]
            ),
            resolve=False,
        )
    observation_space = spaces.Dict(
        {
            "rgb": spaces.Box(0, 255, (224, 224, 3), dtype=np.uint8),
            "semantic": spaces.Box(0, 1, (224, 224, 1), dtype=np.float32),
            "clip_objectgoal": spaces.Box(-np.inf, np.inf, (768,), dtype=np.float32),
            "step_id": spaces.Box(0, 499, (1,), dtype=np.int64),
        }
    )
    create_model = timm.create_model

    def construct_without_download(*args, **kwargs):
        # The complete SigLIP visual weights are already present in ckpt.5.pth.
        kwargs["pretrained"] = False
        return create_model(*args, **kwargs)

    with vendor_directory(), patch("timm.create_model", construct_without_download):
        policy = OVONTransformerPolicy(
            observation_space=observation_space,
            action_space=spaces.Discrete(6),
            transformer_config=OmegaConf.create(settings["transformer_config"]),
            segm_update_config=OmegaConf.create(settings["segm_update_config"]),
            policy_config=OmegaConf.create(settings),
            backbone=settings["backbone"],
            fusion_type=settings["fusion_type"],
            rgb_only=settings["rgb_only"],
            use_prev_action=settings["use_prev_action"],
            use_odom=settings["use_odom"],
            use_vis_query=settings["use_vis_query"],
            use_residual=settings["use_residual"],
            residual_vision=settings["residual_vision"],
            unfreeze_xattn=settings["unfreeze_xattn"],
            segm_mask_input=settings["segm_mask_input"],
            segm_loss_enabled=settings["segm_loss_enabled"],
            # Both GT and detector masks enter the identical external mask path.
            # The caller records the real source and supplies the binary tensor.
            segmentation_source="gt",
        )
    weights = {}
    for name, value in saved["state_dict"].items():
        assert name.startswith("actor_critic."), name
        weights[name.removeprefix("actor_critic.")] = value
    # The author's override uses strict=False for the net. Validate every model
    # key here, including the frozen visual encoder and the auxiliary decoder.
    result = torch.nn.Module.load_state_dict(policy, weights, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    policy = policy.cuda().eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return (
        policy,
        config,
        {
            "strict_load": True,
            "state_tensors": len(weights),
            "state_numel": sum(value.numel() for value in weights.values()),
            "checkpoint_stage_step": saved["extra_state"]["step"],
            "policy_source": str(VENDOR / "ovon/models/transformer_policy.py"),
            "original_eval_config": original_eval,
        },
    )


def start_state(policy, batch=1):
    shape = policy.net.state_encoder._expected_cache_shape
    return torch.zeros(batch, *shape, device="cuda"), torch.zeros(
        batch, 1, device="cuda", dtype=torch.long
    )


class ObjectMaskDetector:
    """Original YOLOE weights, category lists, thresholds and RGB preprocessing."""

    def __init__(self, split="val_unseen"):
        from ultralytics import YOLOE

        if split not in ("train", "val_seen", "val_seen_synonyms", "val_unseen"):
            raise ValueError("Unknown OVON split: " + split)
        settings = json.loads((VENDOR / ("cats_" + split + ".json")).read_text())
        self.categories = settings["cats_text"]
        self.thresholds = settings["conf_per_cat"]
        with (ASSETS / "siglip.pkl").open("rb") as handle:
            self.goal_embeddings = pickle.load(handle)
        self.goal_values = torch.stack(
            [torch.from_numpy(self.goal_embeddings[name]) for name in self.categories]
        ).cuda()
        # MobileCLIPTS looks for its weight in cwd. Keep it with the other
        # baseline assets instead of causing implicit downloads into the repo.
        with vendor_directory(ASSETS):
            self.model = YOLOE(str(ASSETS / "yoloe-v8l-seg.pt")).to("cuda")
            self.model.set_classes(self.categories, self.model.get_text_pe(self.categories))

    def target_class(self, category):
        # The calibrated vocabulary removes redundant synonyms. Match the
        # author's L1-nearest SigLIP goal lookup, including nonliteral aliases.
        goal = torch.from_numpy(self.goal_embeddings[category]).cuda()
        return int(torch.sum(torch.abs(self.goal_values - goal), dim=1).argmin().item())

    @torch.inference_mode()
    def predict(self, rgb, category):
        from ovon.obs_transformers.resize import Resize

        target_class = self.target_class(category)
        threshold_category = self.categories[target_class]
        observation = {"rgb": torch.as_tensor(rgb, device="cuda").unsqueeze(0)}
        observation = Resize((224, 224))(observation)
        result = self.model.predict(
            observation["rgb"].permute(0, 3, 1, 2) / 255,
            save=False,
            imgsz=224,
            device=torch.device("cuda", torch.cuda.current_device()),
            conf=0.01,
            verbose=False,
        )[0]
        mask = torch.zeros((224, 224), device="cuda", dtype=torch.bool)
        if result.masks is not None:
            selected = result.boxes.cls.long() == target_class
            if threshold_category in self.thresholds:
                selected &= result.boxes.conf > self.thresholds[threshold_category]
            selected_masks = result.masks.data[selected]
            if len(selected_masks):
                mask = selected_masks.any(dim=0)
        return mask


def prepare_observation(rgb, mask, goal, frame_id):
    from ovon.obs_transformers.resize import Resize

    observations = {
        "rgb": torch.as_tensor(rgb, device="cuda").unsqueeze(0),
        "semantic": torch.as_tensor(mask, device="cuda").float().unsqueeze(0).unsqueeze(-1),
        "clip_objectgoal": torch.as_tensor(goal, device="cuda").float().unsqueeze(0),
        "step_id": torch.tensor([[frame_id]], device="cuda", dtype=torch.long),
    }
    return Resize((224, 224))(observations)


@torch.inference_mode()
def step(policy, rgb, mask, goal, cache, previous_action, frame_id, *, deterministic=True):
    observations = prepare_observation(rgb, mask, goal, frame_id)
    features, updated, _ = policy.net(
        observations,
        cache,
        previous_action,
        torch.zeros(1, 128, device="cuda"),
        torch.tensor([[frame_id > 0]], device="cuda", dtype=torch.bool),
    )
    distribution = policy.action_distribution(features)
    action = distribution.mode() if deterministic else distribution.sample()
    return action, distribution.probs, updated, policy.critic(features)


if __name__ == "__main__":
    torch.set_num_threads(2)
    _, _, report = load_policy()
    print(json.dumps(report))

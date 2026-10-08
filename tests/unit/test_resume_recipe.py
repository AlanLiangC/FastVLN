import copy
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from streamnav.training.checkpoint import restore_training


@pytest.mark.parametrize("key", ["backbone_lr", "head_lr", "vision_lr", "max_grad_norm"])
def test_resume_rejects_changed_optimizer_recipe_before_loading_state(tmp_path, key):
    with initialize_config_dir(version_base=None, config_dir=str(Path("configs").resolve())):
        config = OmegaConf.to_container(compose(config_name="config"), resolve=True)
    (tmp_path / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    changed = copy.deepcopy(config)
    changed["trainer"][key] += 0.001
    with pytest.raises(ValueError, match=f"recipe changed: {key}"):
        restore_training(tmp_path, None, None, None, [], changed)


def test_resume_rejects_changed_recurrent_output_normalization(tmp_path):
    with initialize_config_dir(version_base=None, config_dir=str(Path("configs").resolve())):
        config = OmegaConf.to_container(compose(config_name="config"), resolve=True)
    (tmp_path / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    changed = copy.deepcopy(config)
    changed["model"]["kda_output_norm"] = True
    with pytest.raises(ValueError, match="configuration changed: kda_output_norm"):
        restore_training(tmp_path, None, None, None, [], changed)


def test_resume_rejects_new_goal_path_in_legacy_checkpoint(tmp_path):
    with initialize_config_dir(version_base=None, config_dir=str(Path("configs").resolve())):
        config = OmegaConf.to_container(compose(config_name="config"), resolve=True)
    (tmp_path / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    changed = copy.deepcopy(config)
    changed["model"]["goal_conditioning"] = "nav_query"
    with pytest.raises(ValueError, match="configuration changed: goal_conditioning"):
        restore_training(tmp_path, None, None, None, [], changed)


def test_resume_rejects_changed_critic_learning_rate(tmp_path):
    with initialize_config_dir(version_base=None, config_dir=str(Path("configs").resolve())):
        config = OmegaConf.to_container(compose(config_name="config"), resolve=True)
    (tmp_path / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    changed = copy.deepcopy(config)
    changed["trainer"]["critic_lr"] = 0.000025
    with pytest.raises(ValueError, match="recipe changed: critic_lr"):
        restore_training(tmp_path, None, None, None, [], changed)


def test_resume_rejects_changed_training_reset_distribution(tmp_path):
    with initialize_config_dir(version_base=None, config_dir=str(Path("configs").resolve())):
        config = OmegaConf.to_container(compose(config_name="config"), resolve=True)
    (tmp_path / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    changed = copy.deepcopy(config)
    changed["trainer"]["curriculum"]["enabled"] = True
    with pytest.raises(ValueError, match="recipe changed: curriculum"):
        restore_training(tmp_path, None, None, None, [], changed)

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from streamnav.data.manifest import file_hash
from streamnav.training.checkpoint import restore_training, validate_resume_configuration
from streamnav.training.ealm import EntropyAdaptiveLossMixer
from streamnav.training.optimizer import OVSegDTLRScheduler, optimizer_epsilon
from streamnav.utils.seed import rng_state


def fork_fixture(tmp_path):
    with initialize_config_dir(version_base=None, config_dir=str(Path("configs").resolve())):
        config = OmegaConf.to_container(compose(config_name="config"), resolve=True)
    source = tmp_path / "source"
    path = source / "checkpoints/update_0000007"
    path.mkdir(parents=True)
    manifest = tmp_path / "train_manifest.json"
    manifest.write_text("{}")
    config["run_dir"] = str(source)
    config["data"]["sources"] = [{"manifest": str(manifest), "weight": 1.0}]
    (path / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    (path / "manifest.json").write_text(
        __import__("json").dumps({"dataset_manifests": {str(manifest): file_hash(manifest)}})
    )
    changed = copy.deepcopy(config)
    changed["run_dir"] = str(tmp_path / "fork")
    changed["trainer"]["fork_recipe_changes"] = ["ealm"]
    changed["trainer"]["ealm"]["minimum_ppo_weight"] = 0.2
    return path, config, changed


def test_recipe_change_requires_explicit_empty_fork(tmp_path):
    path, original, changed = fork_fixture(tmp_path)
    assert set(validate_resume_configuration(path, changed)) == {"ealm"}
    changed["run_dir"] = original["run_dir"]
    with pytest.raises(ValueError, match="new, empty"):
        validate_resume_configuration(path, changed)
    changed["run_dir"] = str(tmp_path / "fork")
    destination = Path(changed["run_dir"])
    destination.mkdir()
    (destination / "train_metrics.jsonl").write_text('{"update": 8}\n')
    with pytest.raises(ValueError, match="new, empty"):
        validate_resume_configuration(path, changed)


def test_fork_never_allows_robot_inputs_or_unlisted_recipe_changes(tmp_path):
    path, _, changed = fork_fixture(tmp_path)
    changed["trainer"]["backbone_lr"] *= 2
    with pytest.raises(ValueError, match="recipe changed: backbone_lr"):
        validate_resume_configuration(path, changed)
    changed["trainer"]["fork_recipe_changes"].append("backbone_lr")
    with pytest.raises(ValueError, match="permit only"):
        validate_resume_configuration(path, changed)


def test_default_zero_floor_is_compatible_with_legacy_resume(tmp_path):
    path, original, _ = fork_fixture(tmp_path)
    original["trainer"]["ealm"]["minimum_ppo_weight"] = 0.0
    original["trainer"]["auxiliary_il"] = {"num_envs": 0}
    assert validate_resume_configuration(path, original) == {}


def test_teacher_execution_change_requires_explicit_empty_fork(tmp_path):
    path, original, _ = fork_fixture(tmp_path)
    changed = copy.deepcopy(original)
    changed["run_dir"] = str(tmp_path / "teacher_fork")
    changed["habitat"]["oracle_execution"] = "collision_safe"
    with pytest.raises(ValueError, match="environment recipe changed: oracle_execution"):
        validate_resume_configuration(path, changed)
    changed["trainer"]["fork_recipe_changes"] = [
        "oracle_execution",
        "filter_blocked_forward_labels",
    ]
    changed["trainer"]["filter_blocked_forward_labels"] = True
    assert set(validate_resume_configuration(path, changed)) == {
        "oracle_execution",
        "filter_blocked_forward_labels",
    }
    changed["run_dir"] = original["run_dir"]
    with pytest.raises(ValueError, match="new, empty"):
        validate_resume_configuration(path, changed)


def test_floor_keeps_ppo_gradient_at_high_entropy_and_survives_ema_restore():
    il = torch.ones(2, requires_grad=True)
    ppo = torch.ones(2, requires_grad=True)
    entropy = torch.ones(2, requires_grad=True)
    parent = EntropyAdaptiveLossMixer()
    parent.observe_entropy(1.0)
    mixer = EntropyAdaptiveLossMixer(minimum_ppo_weight=0.2)
    mixer.load_state_dict(parent.state_dict())
    loss, alpha = mixer(il, ppo, entropy)
    torch.testing.assert_close(alpha, torch.full((2,), 0.8))
    loss.sum().backward()
    torch.testing.assert_close(ppo.grad, torch.full((2,), 0.2))
    assert entropy.grad is None
    for value in [-1, 2, float("nan")]:
        with pytest.raises(ValueError):
            EntropyAdaptiveLossMixer(minimum_ppo_weight=value)


def test_backbone_epsilon_override_preserves_restored_adam_moments(tmp_path):
    path, original, changed = fork_fixture(tmp_path)
    changed["trainer"]["ealm"] = original["trainer"]["ealm"]
    changed["trainer"]["fork_recipe_changes"] = ["backbone_optimizer_eps"]
    changed["trainer"]["backbone_optimizer_eps"] = 1e-6
    parameters = [torch.nn.Parameter(torch.ones(1)) for _ in range(2)]
    optimizer = torch.optim.Adam(
        [
            {"params": [parameters[0]], "role": "backbone", "eps": 1e-5},
            {"params": [parameters[1]], "role": "actor", "eps": 1e-5},
        ]
    )
    sum(p.sum() for p in parameters).backward()
    optimizer.step()
    expected_moment = optimizer.state[parameters[0]]["exp_avg"].clone()
    scheduler = OVSegDTLRScheduler(optimizer, changed["trainer"])
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": {"update": 7},
            "rng": rng_state(),
            "sources": [],
            "update": 7,
        },
        path / "optimizer.pt",
    )
    (path / "dagger_scheduler.json").write_text('{"update": 7}')
    optimizer.state[parameters[0]]["exp_avg"].zero_()
    update = restore_training(path, optimizer, scheduler, SimpleNamespace(update=0), [], changed)
    assert update == 7
    assert [g["eps"] for g in optimizer.param_groups] == [1e-6, 1e-5]
    torch.testing.assert_close(optimizer.state[parameters[0]]["exp_avg"], expected_moment)
    for value in [0.0, -1.0, float("nan")]:
        with pytest.raises(ValueError):
            optimizer_epsilon({"backbone_optimizer_eps": value}, "backbone")

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.PHONY: check unit gpu integration smoke train viewer
check:
	source scripts/env.sh
	ruff check src services tools tests
	ruff format --check src services tools tests
	mypy src
unit:
	source scripts/env.sh
	python -m pytest tests/unit tests/regression -q
gpu:
	source scripts/env.sh
	python -m pytest tests/gpu -q
integration:
	source scripts/env.sh
	STREAMNAV_INTEGRATION=1 python -m pytest tests/integration -q
smoke:
	source scripts/env.sh
	python -m streamnav.training.trainer data=hm3d_v1 eval=hm3d_v1 run_dir=runs/recovery_smoke_20261001 trainer.num_updates=1 trainer.num_envs=2 trainer.rollout_steps=4 trainer.sequence_length=4 trainer.sequence_batch_size=2 trainer.update_epochs=1 trainer.checkpoint_interval=1 trainer.eval_interval=1 trainer.eval_episodes=3 trainer.eval_max_steps=4
train:
	bash scripts/start_eight_gpu_training.sh
viewer:
	bash scripts/start_viewer.sh

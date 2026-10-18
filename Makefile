SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
VIEWER_RUN_DIR ?= $(if $(wildcard runs/qwen35_pointcot_opos_hint/evaluation),runs/qwen35_pointcot_opos_hint,runs/pointcot_grounding_refined)
.PHONY: check unit gpu integration smoke grounding train pipeline viewer
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
	python tools/validate_pointcot.py
integration:
	source scripts/env.sh
	STREAMNAV_INTEGRATION=1 python -m pytest tests/integration -q
grounding:
	bash scripts/start_grounding_training.sh
smoke:
	source scripts/env.sh
	python -m streamnav.training.pointcot_trainer --run-dir runs/pointcot_smoke_$$(date -u +%Y%m%d_%H%M%S) --initialize-from runs/pointcot_grounding_refined/checkpoints/best --updates 2 --rollout-steps 4 --sequence-length 2 --eval-episodes 3 --eval-max-steps 8 --skip-initial-eval
train:
	bash scripts/start_pointcot_training.sh
pipeline:
	bash scripts/start_training_pipeline.sh
viewer:
	python -m http.server 8765 --directory "$(VIEWER_RUN_DIR)"

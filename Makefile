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
	python tools/validate_lightnav_kda.py
integration:
	source scripts/env.sh
	STREAMNAV_INTEGRATION=1 python -m pytest tests/integration -q
smoke:
	source scripts/env.sh
	python -m streamnav.training.lightnav_trainer --run-dir runs/lightnav_smoke_$$(date -u +%Y%m%d_%H%M%S) --updates 2 --num-envs 1 --rollout-steps 4 --sequence-length 2 --eval-episodes 3 --eval-max-steps 8 --skip-initial-eval
train:
	bash scripts/start_lightnav_training.sh
viewer:
	python -m http.server 8765 --directory runs/lightnav_kda_active

#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$STREAMNAV_ROOT"
python - "${1:-runs/streamnav_active/ealm}" <<'PY'
import json,os,signal,sys
from pathlib import Path
root=Path(sys.argv[1]);workers=root/'training_workers.json'
pids=[w['pid'] for w in json.loads(workers.read_text())] if workers.exists() else [int((root/'training.pid').read_text())]
for pid in pids:
    command=Path(f'/proc/{pid}/cmdline')
    if command.exists() and any(token in command.read_bytes() for token in (b'streamnav.training.trainer',b'tools/check_ovsegdt_training.py')):
        os.kill(pid,signal.SIGTERM)
        print(f'Requested checkpoint and graceful stop: {pid}')
PY

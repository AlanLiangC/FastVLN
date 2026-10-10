#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$STREAMNAV_ROOT"
python - "${1:-runs/streamnav_active/ealm}" <<'PY'
import json,os,signal,sys,time
from pathlib import Path
from streamnav.utils.process_health import training_stop_targets
root=Path(sys.argv[1]);workers=root/'training_workers.json'
launcher=int((root/'training.pid').read_text())
records=json.loads(workers.read_text()) if workers.exists() else []
pids=training_stop_targets(records,launcher)
if pids:
    request=root/'stop_request.json'
    temporary=request.with_suffix('.tmp')
    temporary.write_text(json.dumps({'training_pid':launcher,'time':time.time(),'targets':pids}))
    temporary.replace(request)
for pid in pids:
    try:
        os.kill(pid,signal.SIGTERM)
        print(f'Requested checkpoint and graceful stop: {pid}')
    except ProcessLookupError:
        pass
PY

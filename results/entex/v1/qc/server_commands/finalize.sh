#!/bin/bash
set -euo pipefail
cd /home/tuv43532/PangenomeFM
export PATH=/home/tuv43532/miniconda3/envs/pangenomefm-server/bin:$PATH
export PYTHONPATH="$PWD:$PWD/src" GIT_TERMINAL_PROMPT=0
exec > results/entex/v1/server/finalize.log 2>&1
trap 'printf "%s\n" "$?" > results/entex/v1/server/finalize.exit' EXIT
# Keep active jobs on their current source until every long-running shell exits.
for gate in sensitivities p1_full p2_full; do
  while [ ! -f "results/entex/v1/server/${gate}.exit" ]; do sleep 30; done
done
test "$(git branch --show-current)" = codex/canonical-complexity-v2-20260815
mkdir -p server_workspace/entex_result_backup_before_final_sync
for name in p0_exposure_matched_analysis p0_h3k27ac_analysis p0_ctcf_analysis; do
  if [ -d "results/entex/v1/$name" ]; then
    cp -a "results/entex/v1/$name" server_workspace/entex_result_backup_before_final_sync/
  fi
done
git pull --ff-only
if [ "$(cat results/entex/v1/server/sensitivities.exit)" = 0 ]; then
  python -m tasks.entex.sensitivity_report
fi
python - <<'PYCODE'
import json,pathlib,subprocess
root=pathlib.Path('results/entex/v1/server')
status={name:int((root/(name+'.exit')).read_text()) for name in ['sensitivities','p1_full','p2_full']}
(root/'completion_status.json').write_text(json.dumps(dict(exit_codes=status,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),all_experiments_succeeded=all(x==0 for x in status.values())),indent=2)+'\n')
if any(status.values()): raise SystemExit('At least one experiment failed; inspect its log. Code synchronization completed.')
PYCODE

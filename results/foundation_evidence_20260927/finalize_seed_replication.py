"""Instance launcher: finish the already running fixed-seed experiment report."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

from tasks.transfer.frozen_branch_replication_report import build

ROOT = Path('/home/tuv43532/PangenomeFM_readiness_20260927')
BASE = ROOT / 'results/foundation_evidence_20260927'
REPLICATION = Path('/home/tuv43532/PangenomeFM_evidence_report_20260927/results/foundation_evidence_20260927/frozen_branch_seed_replication')
JOURNAL = BASE / 'seed_replication_finalizer_status.json'
OUT = BASE / 'frozen_branch_seed_replication_analysis'
if JOURNAL.exists() or OUT.exists():
    raise FileExistsError('Refusing to overwrite a prior report or finalizer')
record = dict(status='waiting_for_existing_experiment', replication_root=str(REPLICATION),
              report_root=str(OUT), maximum_wait_hours=8,
              implementation_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def save():
    temporary = JOURNAL.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(JOURNAL)


save()
try:
    for attempt in range(960):
        status = json.loads((REPLICATION / 'status.json').read_text())
        if status['status'] == 'failed':
            raise RuntimeError(status.get('error', 'Source replication failed'))
        if status['status'] == 'complete':
            break
        time.sleep(30)
    else:
        raise TimeoutError('Source replication did not finish within eight hours')
    build([BASE / 'composite_biological_analysis',
           REPLICATION / 'seed_314159/biological_analysis',
           REPLICATION / 'seed_20260806/biological_analysis'], OUT)
    record['status'] = 'complete'
except Exception as exc:
    record.update(status='failed', error=str(exc))
    raise
finally:
    save()

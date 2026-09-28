"""Finish supplemental reporting and complete the audited whole-graph NT cache."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.server.prepare_node_sequence_fm_cache import sha256_file
from tasks.transfer.traitgym import write_json

root = Path(__file__).resolve().parents[3]
os.chdir(root)
directory = Path(__file__).parent
plan_path = root / 'configs/whole_graph_nt_completion_20260928.json'
plan = json.loads(plan_path.read_text())
status_path = directory / 'status.json'
if status_path.exists():
    raise FileExistsError('Refusing to overwrite post-campaign receipt')
record = dict(status='waiting', plan=plan, native_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              plan_sha256=sha256_file(plan_path), commands=[])
write_json(status_path, record)
env = dict(os.environ, PYTHONPATH='src:.', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MPLCONFIGDIR='/tmp/pfm-mpl-post')


def execute(command, name):
    record.update(status='running', stage=name)
    record['commands'].append(command)
    write_json(status_path, record)
    with (directory / (name+'.log')).open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


try:
    deadline = time.monotonic() + 12*3600
    while True:
        dependency = json.loads(Path(plan['dependency']).read_text())
        if dependency['status'] in ['failed', 'cancelled']:
            raise RuntimeError('Masked-feature dependency failed; inspect it before consuming GPUs')
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
        usage = {int(i): int(memory) for i, memory in (line.split(',') for line in raw.splitlines())}
        record.update(dependency_status=dependency['status'], gpu_memory_mib=usage,
                      observed_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        write_json(status_path, record)
        if dependency['status']=='complete' and all(usage[i]<256 for i in plan['gpus']):
            break
        if time.monotonic()>deadline:
            raise TimeoutError('Dependency/GPU readiness deadline exceeded')
        time.sleep(60)
    execute([sys.executable, '-m', 'tasks.transfer.masked_junction_reference',
             '--config', 'configs/masked_nt_junction_reference_20260928.json',
             '--masked-report', 'results/foundation_evidence_20260927/masked_feature_full_analysis',
             '--out-dir', 'results/foundation_evidence_20260927/masked_junction_full_reference'], 'junction_reference')
    if shutil.disk_usage(root).free < plan['minimum_free_disk_bytes']:
        raise RuntimeError('Insufficient disk for audited cache completion')
    if sha256_file(Path(plan['existing_cache'])) != plan['existing_cache_sha256'] or sha256_file(Path(plan['graph'])) != plan['graph_sha256']:
        raise ValueError('Original graph or benchmark cache changed')
    execute([sys.executable, '-u', 'scripts/server/complete_node_sequence_fm_cache.py',
             '--full-segments', plan['graph'], '--existing-cache', plan['existing_cache'],
             '--out-dir', plan['output'], '--device', 'cuda', '--batch-size', str(plan['batch_size']),
             '--maximum-new-segments', str(plan['expected_new_segments']), '--shard-gpus', *map(str, plan['gpus']), '--execute'], 'whole_graph_cache')
    completion = json.loads((Path(plan['output'])/'status.json').read_text())
    if (completion['status']!='complete' or completion['target_segments']!=plan['expected_graph_segments']
            or completion['missing_segments']!=plan['expected_new_segments']
            or not completion['serialized_extension_audit']['original_vectors_bitwise_unchanged']):
        raise ValueError('Cache completion differs from audited scope')
    record.update(status='complete', completion=completion['serialized_extension_audit'])
except BaseException as error:
    record.update(status='failed', error=repr(error))
    raise
finally:
    write_json(status_path, record)

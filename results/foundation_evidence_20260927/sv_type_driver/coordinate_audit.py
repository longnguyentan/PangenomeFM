import json
from pathlib import Path
import pandas as pd
from tasks.transfer.sv_types import normalize, validate_padded_vcf
from tasks.transfer.traitgym import verified_fingerprint
root=Path('/home/tuv43532/PangenomeFM_evidence_report_20260927')
base=root/'results/foundation_evidence_20260927'
source=Path('/home/tuv43532/PangenomeFM/server_workspace/data/downstream/sv/hgsvc3/1.0/GRCh38')
qc=json.load(open(base/'sv_type_preparation/qc.json'))
for item in qc['sources']:
    verified_fingerprint(source/Path(item['path']).name,item['sha256'])
data=normalize(pd.read_csv(source/'variants_GRCh38_sv_insdel_HGSVC2024v1.0.tsv.gz',sep='\t'))
audit=validate_padded_vcf(data,source/'variants_GRCh38_sv_insdel_alt_HGSVC2024v1.0.vcf.gz')
(base/'sv_type_driver/coordinate_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))

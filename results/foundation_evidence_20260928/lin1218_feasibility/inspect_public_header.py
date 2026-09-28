from pathlib import Path
import gzip,hashlib,json,requests,time
out=Path('results/foundation_evidence_20260928/lin1218_feasibility');out.mkdir(exist_ok=True,parents=True)
r=requests.get('https://api.github.com/repos/jiadong324/1KG_LongRead_SV/commits/main',timeout=30);r.raise_for_status();commit=r.json()['sha']
r=requests.get(f'https://api.github.com/repos/jiadong324/1KG_LongRead_SV/git/trees/{commit}?recursive=1',timeout=30);r.raise_for_status();tree=r.json()
files=[x['path'] for x in tree['tree'] if x['type']=='blob']
r=requests.get('https://zenodo.org/api/records/22000872',timeout=30);r.raise_for_status();meta=r.json()
selected=next(x for x in meta['files'] if x['key']=='GRCh38_INSDEL_1218_wAF.vcf.gz')
url=selected['links']['self'];header=[];first=None
with requests.get(url,stream=True,timeout=60) as response:
 response.raise_for_status()
 with gzip.GzipFile(fileobj=response.raw) as z:
  for i in range(10000):
   line=z.readline().decode()
   if not line:break
   if not line.startswith('#'):first=line.rstrip().split('\t');break
   header.append(line.rstrip())
columns=next(x for x in header if x.startswith('#CHROM')).split('\t')
record=dict(status='header_and_public_inventory_audited',observed_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
 source_repo='https://github.com/jiadong324/1KG_LongRead_SV',source_commit=commit,repository_files=files,
 zenodo_record='https://zenodo.org/records/22000872',zenodo_created=meta.get('created'),zenodo_publication_date=meta['metadata'].get('publication_date'),
 callset_url=url,callset_expected_bytes=selected['size'],callset_expected_checksum=selected['checksum'],vcf_header_lines=header,
 n_genotype_samples=max(0,len(columns)-9),first_record_metadata=dict(chrom=first[0],position=int(first[1]),id=first[2],info=first[7],reference_length=len(first[3]),alternate_length=len(first[4])),
 full_file_downloaded=False,per_variant_measured_genotyping_outcomes_found=False,
 scope='Public metadata/header inspection only; no claimed full callset QC, donor independence or model evaluation.')
(out/'source_audit.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='vcf_header_lines'},indent=2))
print('\n'.join(x for x in header if x.startswith('##INFO=') or x.startswith('##FORMAT=')))

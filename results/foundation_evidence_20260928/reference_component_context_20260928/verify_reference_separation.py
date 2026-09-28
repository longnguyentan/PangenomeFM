from pathlib import Path
import json,numpy as np,pandas as pd
from graph.slicing import build_global_index,map_links_to_segids
from tasks.entex.prepare import fingerprint
b=Path('/home/tuv43532/PangenomeFM_evidence_report_20260927/results/foundation_evidence_20260927')
g=Path('/home/tuv43532/PangenomeFM/server_workspace/data/processed/hprc_r2_sv')
n=pd.read_csv(g/'full_segments.csv.gz',usecols=['name','SN']);l=pd.read_csv(g/'full_links.csv.gz',usecols=['from_seg','to_seg'])
ix,n=build_global_index(n);s,t=map_links_to_segids(l,ix)
ref=n.SN.astype(str).str.startswith('GRCh38#0#').to_numpy();sn=n.SN.to_numpy()
mask=ref[s]&ref[t]&(sn[s]!=sn[t])
components=pd.read_parquet(b/'reference_component_context_20260928/alternative_components.parquet')
result=dict(status='complete',direct_reference_crosscontig_links=int(mask.sum()),alternative_components_with_multiple_reference_contigs=int(components.n_reference_contigs.gt(1).sum()),
 graph_sha256='e0d832a820969403797662f9af267440599898f068ba9df4069b114d669a3347',links_sha256='7435ac55187d7bca464b961621aada6f8d46e0f7ef900ff7d59d0daee89e2f3b',
 source_component_audit=fingerprint(b/'reference_component_context_20260928/audit.json'),script_sha256=fingerprint(Path(__file__))['sha256'],
 interpretation='Absence of both classes implies that no graph path connects two distinct GRCh38 reference contigs. Graph-derived one-hop context cannot cross between those contigs in this exact graph. This does not address donor overlap or biological label ascertainment.')
(b/'reference_contig_separation_20260928.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

# Frozen downstream transfer, 2026-09-22

## Fixed scope before fitting

Configuration: `configs/downstream_transfer_v1.json`. Strict context and AUPRC
are primary; one-hop is a sensitivity. Retain all outcomes, including null and
negative gains. Reuse the exact HPRC R2 graph, checkpoint, NT revision, folds,
features and logistic classifier used for the manuscript and EN-TEx. No encoder
is trained with downstream labels. Completed EN-TEx CSVs are protected by
`configs/completed_entex_regression_20260922.json` (cached-artifact checks, not
retraining). Unrelated manuscript figure changes are outside this work.

## HG008: correct the proposed target

The original HGSVC probe predicts **insertion versus deletion**, not breakpoint
versus background. Its scores cannot establish breakpoint localization. The
new external adapter preserves INS=1, DEL=0, length >=50 bp, and uses only
V0.5 PASS records explicitly marked `SUBCLONAL=n`, with biological endpoints
inside the clonal benchmark BED. BND/DUP and subclonal calls are excluded with
counts. NIST MD5s are checked before preparation.

Source: [NIST V0.5 release and README](https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data_somatic/HG008/Liss_lab/analysis/NIST_HG008-T_somatic-stvar-CNV_DraftBenchmark_V0.5-20260318/).
PASS now includes subclonal calls, so PASS alone is insufficient.

The original mapper/normalizer is reused, including its legacy inferred
insertion pseudo-end when END is absent. That pseudo-end is a representation
convention, not a second biological insertion breakpoint. The primary nearest
mapping allowance is the original 10 kb; overlap-only is separately reported.

Fitted old logistic probes were not saved. Reconstruct them from the **original
HGSVC train and validation chromosomes only**, validate held-out original
predictions against cached manuscript outputs (AUPRC tolerance 1e-4), and only
then score HG008 on the corresponding excluded chromosomes. HG008 labels never
enter fitting, calibration or threshold selection. Stop on regression failure.
Compare C+S and C+S+T on identical examples. The 69 eligible variants (48 DEL,
21 INS) from one genome make this a small descriptive external transfer study.
It cannot establish general cancer prediction or localization. Do not report
top-1%/5% breakpoint enrichment for this incompatible target.

## GTEx: matched posterior discrimination feasibility

Source: [GTEx downloads](https://gtexportal.org/home/downloads/adult-gtex/overview),
V10 SuSiE archive and version-matched GENCODE39 gene GTF:

- https://storage.googleapis.com/adult-gtex/bulk-qtl/v10/susie-qtl/GTEx_v10_SuSiE_eQTL.tar
- https://storage.googleapis.com/adult-gtex/references/v10/reference-tables/gencode.v39.GRCh38.genes.gtf

The compact release contains credible-set variants, not the complete tested
universe. Define high PIP >=0.9 and low PIP <=0.01 **among released variants**.
Do not label absent variants negative or equate low posterior with proven
noncausality. Restrict to SNVs, exclude multi-allelic and conflicting genomic
positions, match 1:1 without replacement by gene, chromosome, MAF bin and
absolute TSS-distance bin. Exact gene IDs include versions. VCF-style variant
positions and GTF positions are one-based; graph intervals are zero-based.
Local LD matching is unavailable and is not claimed. Allele effect direction
requires an allele-aware representation and is not inferred from locus T.

Whole_Blood, Liver and Brain_Cortex were chosen for biological breadth before
fitting. Strict matching leaves respectively 92, 34 and 42 unique loci. These
are a **feasibility failure for a powered benchmark**, not a negative model
result. Do not loosen matching based on model outcomes. Investigate all-tested
fine-mapping resources before a full eQTL experiment. The reusable probe has an
opt-in eQTL adapter with within-chromosome T permutation and training-label
permutation; existing EN-TEx behavior is unchanged when the flag is absent.

## Path task audit

Existing HPRC path extraction, donor splits and branch-choice builders should
be reused. The path-rich GBZ/full-resolution segment universe differs from the
manuscript SV-scale graph. A historical chr22 attempt using manuscript segment
IDs had 100% missing handles; the full-resolution run had zero missing handles.
Never join these node IDs directly. Donor-label holdout on a graph containing
those donors is transductive, not donor-excluded pretraining. Donor-excluded
graphs and a verified node/representation correspondence remain prerequisites.
No architecture rewrite or replacement graph is part of this transfer campaign.

## Commands

Run from the repository with `PYTHONPATH=src:.`; use the existing
`pangenomefm-server` environment on the server.

```bash
python -m tasks.transfer.hg008 --source-dir data/downstream_v2/sources/hg008_v05 --out-dir data/downstream_v2/v1/hg008
python -m tasks.transfer.eqtl --archive data/downstream_v2/sources/gtex_v10/GTEx_v10_SuSiE_eQTL.tar --gtf data/downstream_v2/sources/gtex_v10/gencode.v39.GRCh38.genes.gtf --out-dir data/downstream_v2/v1/eqtl
python scripts/server/prepare_sv_breakpoint_examples.py --normalized-variants data/downstream_v2/v1/hg008/normalized_variants.csv.gz --full-segments server_workspace/data/processed/hprc_r2_sv/full_segments.csv.gz --out-dir data/downstream_v2/v1/hg008_mapping --maximum-nearest-distance 10000
python -m tasks.transfer.external_sv --external-examples data/downstream_v2/v1/hg008_mapping/sv_breakpoint_examples.csv.gz --out-root results/downstream_v2/v1/hg008_smoke --fold fold_a --seed 42 --context strict
python -m pytest -q tests/test_transfer.py tests/test_entex.py tests/test_sv_frozen_probe.py tests/test_prepare_sv_breakpoint_examples.py
```

Resource downloads and prepared Parquet files are excluded from Git. Source
fingerprints, compact QC, summaries, tests, figures and commands are retained.
Results status and paths will be updated after server validation.

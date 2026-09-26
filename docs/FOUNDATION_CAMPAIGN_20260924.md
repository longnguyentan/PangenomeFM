# Structural transfer campaign, 24 September 2026

The maintained task list is [FOUNDATION_CAMPAIGN_CHECKLIST.md](FOUNDATION_CAMPAIGN_CHECKLIST.md).
Preserve the canonical v1 model and completed EN-TEx outputs. This campaign
tests reusable representations; null outcomes remain part of the report.

## Completed structural analysis

All 30 archived HGSVC probe runs were scored without refitting, using seven
feature combinations. The label is INS=1 versus DEL=0 among mapped variants,
not breakpoint presence. Class-only subsets therefore receive recall, not AP
or AUROC. Frequency/length bins are inherited unchanged. Complexity is the
maximum native category at the two legacy endpoint coordinates; start-only
is a separately retained sensitivity.

| Native complexity | Strict ΔAP (95% CI) | One-hop ΔAP (95% CI) |
|---|---:|---:|
| Low | 0.04193 (0.03315, 0.05244) | 0.04986 (0.04180, 0.06160) |
| Medium | 0.03803 (0.03539, 0.04034) | 0.04740 (0.04351, 0.05069) |
| High | 0.03021 (0.02821, 0.03217) | 0.03362 (0.03042, 0.03635) |

High-minus-low gain is −0.01171 (−0.02300, −0.00192) strict and −0.01624
(−0.02671, −0.00918) one-hop. Thus the proposed stronger high-complexity
gain is **not supported**. The same ordering is visible in AUROC and
prevalence-normalized AP, `(AP − prevalence)/(1 − prevalence)`; it is not
appropriate to select a different complexity definition after observing this.

Intervals use the existing 10,000-draw paired chromosome-fold/seed bootstrap.
Five seed-averaged folds yield a minimum two-sided exact sign-flip p of 0.0625;
the strict complexity interaction has p=0.125 and one-hop p=0.0625.
Bootstrap intervals and these coarse tests answer different uncertainty
questions. Neither removes dependence from overlapping cross-validation
training sets. BH q-values are provided within family/context/metric.
Do not equate a pointwise interval excluding zero with multiplicity-adjusted
evidence of mechanism.

Outputs: `results/foundation_campaign/20260924/sv_strata/` and
`structural_report/`. The latter contains absolute seven-feature metrics,
paired secondary metrics, and PDF/SVG/300-dpi PNG figures.

## Completed donor/haplotype analysis and its limits

The original 65-sample HGSVC VCF supplies phased carrier assignments. Missing
alleles remain unknown. All 30 original chromosome-held-out probe runs are
stratified by donor and haplotype without refitting. The alias audit confirms
HG002/NA24385 is one individual ([NCBI BioSample](https://www.ncbi.nlm.nih.gov/biosample/SAMN03283347/)).
The overlap list contains HG00733, HG02818, NA19036, NA19240 and NA24385.

Across the 60 nonoverlapping donors, the donor-macro topology gain is
0.03673 (0.03391, 0.03946) strict and 0.04303 (0.03928, 0.04702) one-hop.
These are **donor-stratified chromosome-held-out** results: the downstream
probe was trained on pooled HGSVC labels from other chromosomes, including
the same donors. They are not donor-held-out probe validation or personalized
embeddings. Shared variants correlate donor outcomes; donors are not treated
as 60 independent replicates. Confidence intervals resample folds and seeds.
No within-distribution donor-specific comparator exists here, so no misleading
performance-retention ratio is constructed.

Outputs: `results/foundation_campaign/20260924/donor_sv/` and
`structural_report/donor_{absolute,paired}.csv`.

## HG008 regression stop

24/30 jobs passed the fixed original-probe AP tolerance; six stopped before
external completion. The C+S original scores reproduce to numerical precision.
C+S+T AP differences in failed runs range in absolute value from approximately
0.00010 to 0.00045. A fresh execution of the original SV pipeline on
fold D/42/strict also differed from the archived topology-probe scores
(AP difference −0.000557). The problem is therefore not resolved by bypassing
the EN-TEx cache. Further extraction/optimizer reproducibility diagnosis is
required. Do not relax the prespecified 1e-4 gate or summarize only successful
runs as the complete HG008 experiment. Existing results remain unchanged.

## Scaling status and reproduction

Four 100-epoch/patience-20 pilots completed on fold A/42/strict, one per data
fraction. These use unchanged architecture/objective and nested training
windows from the exact graph. Validation and test windows remain fixed.
All scales use the current explicit random seeding and canonical conflict
exclusion; the 100% model is newly trained for a fair control. Single-fold
results do not support a multi-fold CI or a scaling-law claim.

From the server checkout, with its existing environment activated:

```bash
export PYTHONPATH=src:.
export PANGENOMEFM_DATA_ROOT=server_workspace/data
export PANGENOMEFM_RESULTS_ROOT=server_workspace/results
python -m tasks.transfer.scaling \
  --manifest server_workspace/data/benchmarks/hprc_r2_pretrain_5mb_paired/manifest.csv \
  --out-dir data/foundation_campaign/20260924/scaling_matrix \
  --checkpoint-root results/foundation_campaign/20260924/scaling_full \
  --folds fold_a fold_b fold_c fold_d fold_e --seeds 42 314159 20260806 \
  --contexts strict 1hop --balanced-stages \
  --completed-execution-root results/foundation_campaign/20260924/scaling_full_execution
# Launch batch_0..batch_3 separately on GPUs 0..3 using the existing executor.
CUDA_VISIBLE_DEVICES=0 python scripts/server/run_foundation_model_roadmap.py \
  --config data/foundation_campaign/20260924/scaling_matrix/roadmap.json \
  --result-root results/foundation_campaign/20260924/scaling_matrix_execution/gpu_0 \
  --stage batch_0 --execute --allow-gpu
```

Successful pilots are reused only when the command, manifest contents and
persisted checkpoint agree. The generator records hashes of reused checkpoints.
Never overwrite a completed output root.

Recreate the structural report locally or on the server:

```bash
PYTHONPATH=src:. python -m tasks.transfer.report \
  --strata results/foundation_campaign/20260924/sv_strata \
  --donors results/foundation_campaign/20260924/donor_sv \
  --out-dir results/foundation_campaign/20260924/structural_report
PYTHONPATH=src:. python -m pytest -q tests/test_structural_campaign.py tests/test_transfer.py
```

All biological encoders stay frozen. Path-aware work remains a distinct
experimental track, blocked on compatible node/path definitions; no canonical
graph release is replaced.


## Update: 26 September 2026

All **120/120 pretraining runs** completed. The intrinsic audit verified identical
held-out and validation candidate identities/labels at all four fractions within
each fold/seed/context, identical training arguments except manifest/output path,
and the same 52,033 trainable parameters. No large pretrained encoder was changed.

| Requested training windows | Strict pooled-fold AP | One-hop pooled-fold AP |
|---|---:|---:|
| 12.5% | 0.918359 | 0.963541 |
| 25% | 0.929192 | 0.972073 |
| 50% | 0.937487 | 0.984692 |
| 100% | 0.941961 | 0.990006 |

Paired 100%-minus-12.5% gains are 0.023602 [0.021548, 0.025747] strict and
0.026464 [0.025285, 0.027924] one-hop. This is an **intrinsic reconstruction**
result, not yet downstream scaling evidence. Window counts are rounded upward
within chromosomes and unusable slices retain the trainer's exclusions. Actual
counts and epochs are in `scaling_intrinsic/per_run.csv`. Its
`training_candidate_count` counts available positive+negative candidates in
training windows before their internal split; it is not optimizer-update count.
The architecture and epoch/patience limits are fixed, but effective compute is
not: mean completed epochs range from 77.5 to 99.7 across fraction/context groups.
Do not call this haplotype-diversity scaling, compute-matched scaling, or a law.

```bash
PYTHONPATH=src:. python -m tasks.transfer.scaling_intrinsic \
  --checkpoint-root results/foundation_campaign/20260924/scaling_full \
  --out-dir results/foundation_campaign/20260924/scaling_intrinsic
# An existing completed output directory is never overwritten.
```

Frozen biological evaluation is running: four fractions × 30 checkpoint runs ×
three tasks (SV, cCRE, EN-TEx CTCF) = **360 task evaluations**. The 12 completed
pilot evaluations are reused only after checkpoint/audit verification. Eight
CPU batches use the same device as the pilots; no biological labels update the
encoders. The plan is `configs/scaling_probe_jobs_v1.json`:

```bash
# Launch batch_0 through batch_7 in separate sessions; four BLAS/OpenMP threads each.
PYTHONPATH=src:. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python scripts/server/run_foundation_model_roadmap.py \
  --config configs/scaling_probe_jobs_v1.json \
  --result-root results/foundation_campaign/20260924/scaling_probe_execution/batch_0 \
  --stage batch_0 --execute
```

### Manuscript interval correction

The earlier reconstruction plot averaged window AP for its points but
bootstrapped equally weighted chromosome means for intervals. The revised
`manuscript/revision_20260924/build_evidence.py` resamples chromosomes with their
window counts retained. All ten headline means remain identical within 1e-12;
a synthetic unequal-window-count test guards the distinction. This correction
does not change original downstream probes or the four cached regression anchors.
The working LaTeX includes completed EN-TEx, structural and intrinsic scaling
findings while explicitly withholding unfinished biological scaling/HG008 claims.

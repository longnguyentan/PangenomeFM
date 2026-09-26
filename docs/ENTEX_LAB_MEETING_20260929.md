# EN-TEx biological transfer — team brief for Tuesday, 29 September 2026

Prepared 26 September 2026. This document separates completed biological
results, exploratory follow-ups, and experiments still running. The maintained
project-wide checklist is [here](FOUNDATION_CAMPAIGN_CHECKLIST.md).

## 1. What we can currently say

The original EN-TEx programme is complete: **330 fold/seed/context fits** covering
AS-prone cCREs, three predefined cCRE sensitivities, five active/repressed enhancer
tissues, and CTCF/H3K27ac SNVs. Global cCRE and enhancer gains are inconclusive.
The strongest current regulatory evidence is a **small, task-dependent gain in
SNV allelic-imbalance prediction**, especially for the one-hop context.

This supports investigating reusable topology information; it does not establish
universal regulatory improvement, personalized allelic prediction, or a general
foundation-model claim. All null and negative outcomes remain in the report.

The additional analyses below were defined on 26 September **after** inspecting
the primary results. They are exploratory follow-ups, not retrospectively
prespecified confirmatory tests. Their rules were saved before calculating their
scores in [`entex_meeting_20260929.json`](../configs/entex_meeting_20260929.json).

## 2. Scientific question and unchanged methods

**Does self-supervised pangenome connectivity add useful information beyond
genomic coordinates and sequence?** The primary contrast is
`ΔT = AUPRC(C+S+T) − AUPRC(C+S)` on the same examples and fold/seed pairs.

- **C:** existing segment features: log1p reference offset, log1p length,
  orientation. The definition is unchanged for EN-TEx.
- **K:** existing normalized mono/di/tri-nucleotide composition.
- **S:** frozen original Nucleotide Transformer v2 50M segment embeddings.
- **T:** frozen original PangenomeFM segment embeddings from the exact HPRC R2
  SV graph. Neither encoder receives EN-TEx label gradients.
- Only the existing standardized logistic probe is fitted. Five chromosome folds,
  three seeds, strict and one-hop contexts; all occurrences of a locus stay in one
  chromosome partition. Calibration and threshold selection use validation only.
- AUPRC is average precision; its random-ranking reference is positive prevalence.
  AUROC is also reported. Normalized AP is `(AP − prevalence)/(1 − prevalence)`.
- Intervals use the existing paired hierarchical fold/seed bootstrap (10,000
  draws). They are pointwise, not adjusted for this expanded family of analyses.
  Five folds and overlapping training sets limit inferential precision.

EN-TEx supplies accessible/testable measurements and significance calls; an
unmeasured locus is never a negative. The high-confidence file is listed under
the AS catalogue, while the full file is the accessible heterozygous-SNV call
set. Our actual schema audit found the supplied high-confidence file is RNA-only.
Sources: [official data portal](https://entex.encodeproject.org/main.html),
[EN-TEx paper](https://pubmed.ncbi.nlm.nih.gov/37001506/).

## 3. Completed primary results

Each row below summarizes 15 fold/seed pairs. The P1 result is the equal-weight
macro-average over the five tissues selected by sample counts before fitting.
Intervals apply to the paired gain, not to a difference of independent intervals.

| Task | Context | AP C+S | AP C+S+T | ΔT [95% CI] |
|---|---|---:|---:|---:|
| AS-prone cCRE | strict | 0.151416 | 0.151625 | +0.000208 [−0.000317, +0.000746] |
| AS-prone cCRE | one-hop | 0.151416 | 0.150762 | −0.000654 [−0.002092, +0.000466] |
| Active/repressed distal enhancer, tissue macro | strict | 0.569931 | 0.570820 | +0.000889 [−0.000389, +0.002326] |
| Active/repressed distal enhancer, tissue macro | one-hop | 0.569931 | 0.570735 | +0.000805 [−0.001531, +0.002467] |
| CTCF SNV AS | strict | 0.061559 | 0.063033 | +0.001473 [+0.000096, +0.003549] |
| CTCF SNV AS | one-hop | 0.061559 | 0.063805 | +0.002245 [+0.000958, +0.003482] |
| H3K27ac SNV AS | strict | 0.049680 | 0.050373 | +0.000693 [−0.000567, +0.001920] |
| H3K27ac SNV AS | one-hop | 0.049680 | 0.053018 | +0.003338 [+0.000120, +0.007962] |

Source: [`main_comparisons.csv`](../results/entex/v1/final_report/main_comparisons.csv).

P0 has 250,722 loci, 28,092 positive (11.20%). The full accessible SNV source
supplies 2,391,015 CTCF measurements at 596,650 loci (3.5094% positive), and
3,168,525 H3K27ac measurements at 713,418 loci (2.3618% positive). These are
global source prevalences; the summaries use the actual held-out-fold prevalence.
Do not compare AP magnitudes across these different tasks as a model ranking.

### Exposure-matched and assay-specific cCRE sensitivities

| Sensitivity | Strict ΔT [95% CI] | One-hop ΔT [95% CI] |
|---|---:|---:|
| Exact measurement-count/chromosome matching | +0.001482 [+0.000098, +0.002925] | +0.002002 [+0.000844, +0.003278] |
| H3K27ac-only cCRE | +0.000584 [+0.000084, +0.001065] | +0.000612 [−0.000533, +0.001689] |
| CTCF-only cCRE | +0.000261 [−0.000370, +0.000952] | +0.000725 [+0.000217, +0.001301] |

The matched set has 50% prevalence by design. Its AP cannot be compared directly
with the original 11.20%-positive population. Moreover, its C-only AP (0.537015)
exceeds C+S+T (0.535099 strict / 0.535618 one-hop): a positive incremental ΔT
does not mean the largest feature set is the best overall classifier.
Source: [all predefined comparisons](../results/entex/v1/p0_sensitivity_report/all_predefined_comparisons.csv).

## 4. New follow-ups for the meeting

### A. Equal-locus-weight SNV evaluation

**Question:** do frequently measured loci dominate the CTCF/H3K27ac result?
Keep every original per-measurement label and held-out prediction, but give each
measurement weight `1 / number of measurements at its locus` within the assay.
Each locus then contributes total weight one. Recompute weighted AP, AUROC,
prevalence, and normalized AP for C+S and C+S+T. This changes evaluation only;
it does not claim that the original classifier was trained with equal locus weight.

### B. Donor and tissue consistency

**Question:** is the result spread across donors/tissues or concentrated in a few?
Stratify the saved chromosome-held-out predictions by donor and tissue, retaining
every group with at least 200 measurements, 10 positives and 10 negatives in
**each of the five held-out folds**. Eligibility uses counts/labels only, never
model performance. Report every eligible group and equal-weight donor/tissue
macro summaries. CTCF eligibility includes all four donors and all 30 tissues.

These are donor/tissue-stratified evaluations of the existing pooled model.
They are **not unseen-donor or unseen-tissue validation**. A donor may have
measurements in both training and test chromosomes.

Results for A/B are being computed from all 60 original SNV prediction runs.
Completed values and figure links will be added after both assay audits finish.

### C. RNA allele-specific-expression SNV task

**Question:** does the same frozen representation transfer to transcript-level
allelic imbalance, beyond chromatin binding/mark assays?

The supplied high-confidence RNA file yields:

| QC item | Observed |
|---|---:|
| Informative measurements | 1,640,580 |
| Unique loci | 466,867 |
| AS positive / negative | 23,546 / 1,617,034 |
| Positive prevalence | 1.4352% |
| Identical duplicate records removed | 0 |
| Mapped to canonical HPRC segments | 466,867 / 466,867 (100%) |
| Exactly one containing segment | 466,867 (100%) |
| Joint C/K/S/T coverage in smoke run | 100% |

The input SHA256 is
`01b6b7d27446010e214a8cdf1569312cb253c6bedc064a942554b9e28f0927e1`.
The original calls define positives and measured negatives. All seven feature
sets use the same locus universe. No model or graph resource was replaced.

Status: preparation/mapping/feature checks passed; the fold-A/42/strict smoke
fit is running. Full 30-run results are not yet claimed. The task predicts
per-measurement imbalance from a static locus representation, not its direction,
the causal allele, or a donor-specific sequence effect. The high-confidence
catalogue selection also differs from the full accessible ChIP-seq source.

## 5. What belongs in the manuscript

- **Main text:** concise complete EN-TEx comparison, including inconclusive P0/P1;
  modest SNV gains and their limits. Prefer “topology-native self-supervised
  learning on human pangenome graph connectivity” with frozen biological reuse.
- **Supplement:** all seven features, exposure/assay sensitivities, complexity
  strata, donor/tissue and equal-locus analyses, QC and per-run tables. RNA is an
  exploratory extension until the full matrix has completed and been audited.
- **Keep separate:** intrinsic reconstruction scaling versus biological scaling;
  donor-stratified versus donor-held-out testing; insertion/deletion classification
  versus detecting SV breakpoints. Current complexity prevalence gradients alone
  are not evidence that learned topology helps most in complex regions.
- **Do not claim:** “first pangenome foundation model,” universal EN-TEx improvement,
  personalized haplotype inference, or general cancer prediction from HG008.

## 6. Remaining programme and blockers

| Item | Current evidence / next step |
|---|---|
| Biological pretraining scaling | 120 pretraining runs complete; 360 frozen task evaluations running. Audit exact prediction identities/labels and then summarize, including native SV complexity. |
| HG008 historical-probe replay | 24/30 passed; six fail the unchanged 1e-4 AP gate. Numerical diagnostics are complete: identical one-thread fits repeat exactly; four-thread fits differ. Original fitted probes were not saved, so exact historical replay remains unresolved. No gate relaxation or selected-run full summary. |
| Path auxiliary objective and controls | Canonical SV graph has no paths; full-resolution path node IDs do not match its segments. Requires an explicit compatible experimental representation before training; no silent graph substitution. |
| Strict unseen-donor / population transfer | Requires donor-excluded graph construction and audited donor/ancestry metadata. Hiding path labels does not remove donor-contributed topology. |
| Paired haplotype ASE / methylation | Existing runners need matched donor-level labels and compatible haplotype embeddings. The new RNA locus task does not satisfy this distinct requirement. |
| GTEx and broader functional panel | Compact credible sets leave only 92/34/42 matched blood/liver/cortex loci; insufficient for the proposed full benchmark. More suitable labels/resources are needed. |
| Stronger sequence / competing-model baselines | Require versioned, task-faithful embeddings on the same examples; not implemented by relabeling an unrelated task. |
| Final manuscript administrative fields | Funding, author contributions and acknowledgements require author-provided facts. |

The numerical HG008 diagnostic used unchanged cached features within its own
repeats, not a verified historical feature-matrix snapshot. Its C+S one-thread
refit also shifts by +0.000438 AP. Hence an apparent passing topology refit under
one numerical setting is not evidence that the historical reconstruction is fixed.
Full evidence: `results/downstream_v2/v1/qc/numerical_diagnostics/`.

## 7. Reproduction and artifacts

On the existing Temple checkout, activate `pangenomefm-server`, then run from the
repository root. Outputs must be new; completed primary results are preserved.

```bash
export PYTHONPATH=src:.
# Preparation can run locally; transfer its audited compact directory to the server.
python -m tasks.entex.snv --source data/hetSNVs_high-confidence_AS.tsv \
  --assays rna --out-dir data/entex/meeting_20260929/rna
bash scripts/server/run_entex_meeting_20260929.sh rna-smoke
# Only after the smoke audit reports complete:
bash scripts/server/run_entex_meeting_20260929.sh rna-matrix
# Re-evaluate saved primary predictions; no probe or encoder refit:
bash scripts/server/run_entex_meeting_20260929.sh followups
```

- Primary EN-TEx: `results/entex/v1/final_report/`; full command guide:
  [ENTEX_EXPERIMENTS.md](ENTEX_EXPERIMENTS.md).
- Follow-ups: `results/entex/meeting_20260929/followups/{ctcf,h3k27ac}/`:
  `per_run.csv`, `summary.csv`, `paired_gains.csv`, eligibility and prediction audits.
- Meeting figures: `results/entex/meeting_20260929/report/` (PDF/SVG/300-dpi PNG).
- RNA server fits: `results/entex/meeting_20260929/{rna_smoke,rna}/`;
  full analysis, once complete: `results/entex/meeting_20260929/rna_analysis/`.
- RNA preparation/mapping: `data/entex/meeting_20260929/rna/` (large data stay out of Git).
- Existing manuscript working source: `manuscript/revision_20260924/`.

## 8. Suggested 30-minute discussion

1. **5 min:** agree on the task definitions, prevalence and frozen-reuse question.
2. **10 min:** examine the complete primary/sensitivity table, including null results.
3. **10 min:** review equal-locus weighting, all donor/tissue strata, and RNA status.
   Check whether the effect survives a change in the evaluation population.
4. **5 min:** decide what belongs in the main text and which distinct path/donor
   experiments justify additional data/model work. Do not select assays or bins
   solely because they yield larger gains.

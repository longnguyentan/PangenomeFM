# PangenomeFM: project status and decision roadmap

Updated 27 September 2026, 19:23 UTC / 15:23 EDT. This consolidates the recent
work rather than treating each added experiment as a separate project.

**Main conclusion:** the core EN-TEx programme and the pretraining-window scaling
study are complete. We have reproducible, task-dependent frozen-transfer results.
We have **not** established a superior v2 model or broad foundation-model
generalization. The immediate priority is to finish and audit the trained/random
controls and correct the v2 objective before increasing training scale.

**Freshness:** the latest server observations were earlier on 27 September,
before SSH expired. Detached jobs were launched successfully, but their current
completion state is unverified. Counts below are the last observed counts,
not an estimate of how far the server has progressed since then.

## 1. What changed over the last few days

| Period | Work accomplished |
|---|---|
| Earlier baseline, retained throughout | Completed core EN-TEx results and exact-resource mapping; locked manuscript regression anchors. |
| 24 September | Added structural subgroup and donor/haplotype analyses; clarified that the SV target is INS versus DEL; implemented nested pretraining-window scaling and external-transfer checks. |
| 26 September | Audited scaling targets, uncertainty and numerical reproducibility; completed the scaling programme; added RNA and measurement-weighting follow-ups and the lab-meeting brief. |
| 26–27 September | Reviewed the proposed v2 branch; corrected topology-control and candidate-construction issues; completed exploratory one-fold H/random controls; launched the full controls and three new EN-TEx assay matrices. |
| 27 September, latest implementation | Audited native v2 loading and masking; found a one-hop endpoint-balance defect, implemented a whole-group filtering correction, preserved the failed audit, and passed 300 tests. Corrected server rerun awaits authentication. |

Some commits use UTC and others EDT. The detailed Git history provides exact
times; this table groups related work rather than asserting completion times
for every individual server job.

## 2. Completed experiments and analyses

Here, a *run* means a task/fold/seed/context job. A job evaluates several feature
sets. Prediction reanalyses are not counted as newly fitted biological models.
All EN-TEx biological tasks retain the original frozen PangenomeFM and frozen
Nucleotide Transformer encoders; only the existing downstream probe is fitted.

| Workstream | Completed scope | Main outcome |
|---|---|---|
| Manuscript regression | Four cached cCRE/SV comparisons | Original gains reproduced exactly; this is a cached-result check, not historical retraining. |
| P0: AS-prone cCRE | Preparation, mapping, 30 runs, seven-feature summaries | Global topology gain inconclusive. |
| P0 sensitivities | Exposure-matched, H3K27ac-only and CTCF-only; 90 runs | Some small positive increments; matched-set C-only remains stronger than the largest feature set. |
| P0b: complexity | Within-stratum comparisons, prevalence, AP/AUROC/normalized AP | Completed; prevalence differences alone are not a learned-topology result. |
| P1: active/repressed distal enhancers | Five tissues, 150 runs | Tissue-macro topology gain inconclusive. Registry mismatch was resolved before fitting. |
| P2/P3: CTCF and H3K27ac SNVs | Full accessible source, 60 runs | Small task/context-dependent gains; proper chromosome grouping retained. |
| Core EN-TEx total | **330 runs** | Complete QC, paired summaries, sensitivity analyses and figures. |
| RNA allelic imbalance | **30 additional runs** | Complete; both global topology-gain intervals cross zero. |
| SNV weighting/donor/tissue follow-ups | Replay of **60 existing prediction runs** | Equal-locus weighting weakens CTCF; H3K27ac remains positive. Donor strata are not donor-held-out fitting. |
| SV mechanism/subgroup analysis | All 30 archived SV runs, original feature combinations | Gain remains positive, but is smaller at high than low graph complexity. |
| Donor/haplotype-stratified SV analysis | All 30 archived runs | Positive stratified gains; no strict unseen-donor claim. At least five graph/evaluation donor overlaps identified. |
| Pretraining-window scaling | **120 pretraining runs** at 12.5/25/50/100% windows | Intrinsic scores increase, but the v1 masking shortcut limits interpretation. |
| Biological scaling | **360 frozen task evaluations**, including SV complexity | Exact baseline invariance and prediction replay passed. Gains depend on task and graph context. |
| Exploratory H/random controls | Eight biological evaluations on fold A, seed 42 | Random embeddings explain substantial utility; results differ by task/context. No multi-fold inference from this pilot. |

### Selected completed biological results

`ΔT = AP(C+S+T) − AP(C+S)`. Values are absolute AP differences, not relative
percentage improvement. Intervals are the existing paired fold/seed bootstrap
intervals; they are pointwise exploratory intervals, not family-wide adjusted
evidence. All five chromosome folds have already been inspected for v1.

| Task/analysis | Strict ΔT [95% CI] | One-hop ΔT [95% CI] | Reading |
|---|---:|---:|---|
| P0 AS-prone cCRE | +0.000208 [−0.000317, +0.000746] | −0.000654 [−0.002092, +0.000466] | Inconclusive globally |
| P1 enhancer, tissue macro | +0.000889 [−0.000389, +0.002326] | +0.000805 [−0.001531, +0.002467] | Inconclusive globally |
| CTCF SNV, original weighting | +0.001473 [+0.000096, +0.003549] | +0.002245 [+0.000958, +0.003482] | Small positive increments |
| CTCF SNV, equal locus weight | +0.000183 [−0.000542, +0.000955] | +0.000493 [−0.000343, +0.001323] | Sensitivity becomes inconclusive |
| H3K27ac SNV, equal locus weight | +0.001990 [+0.001105, +0.002731] | +0.002206 [+0.000653, +0.004192] | Most consistent completed EN-TEx extension evidence |
| RNA imbalance | +0.001568 [−0.000075, +0.004401] | +0.001465 [−0.000218, +0.004303] | Inconclusive globally |

Full values, sample sizes, original H3K27ac weighting, all sensitivities and
figures: [EN-TEx team brief](ENTEX_LAB_MEETING_20260929.md).

The original cached SV topology gains are +0.034246 strict and +0.040090
one-hop; original cCRE gains are +0.004008/+0.003406. Their interpretation now
needs the full H/random controls. In scaling, 100% versus 12.5% training windows
improves strict SV AP by +0.003482 but reduces one-hop SV AP by −0.001510.
More reconstruction training therefore does not universally improve transfer.

## 3. Launched jobs: collect and audit before interpreting

| Job | Last observed state | What remains |
|---|---|---|
| Full trained/random/H controls | 12/15 fold/seed jobs complete = 96/120 evaluations; two final-fold jobs running at that check | Verify all 120 evaluations; replay saved predictions; compare identical targets and baseline scores; generate full paired estimates. |
| ATAC/H3K4me3/H3K27me3 EN-TEx extension | All three smoke runs passed; 2/90 full-panel jobs complete, no recorded failures at that check | Verify 90 runs, all assays/contexts, complexity, equal-locus and donor/tissue results, intervals and multiplicity sensitivity. |
| H/random reporting process | Launched with a bounded dependency wait | It reports only after complete upstream receipts; inspect its status and outputs after reconnecting. |

The additional panel was fixed before its assay scores were inspected. All
three assays map completely to the canonical graph and have full C/K/S/T
coverage in the smoke runs. All three single-fold smoke gains were negative;
the full panel retains every assay regardless. These smoke results are not
substitutes for the full matrix.

## 4. Model-development state

Definitions: **T** is the trained frozen topology encoder, **R** the same encoder
architecture with random weights, and **H** simple label-free graph statistics.
The critical question is whether learned T adds information beyond C+S+H and
beyond R, not merely whether adding more graph-derived features raises AP.

| Component | Status | Boundary of the evidence |
|---|---|---|
| Canonical v1 PangenomeFM | Trained; frozen reuse completed | Existing biological scores are measured results, but intrinsic reconstruction has a masking-related degree cue. |
| Handcrafted H controls | Implemented, corrected and tested | Fixed the capped-eight branching-distance implementation; full downstream matrix still requires collection/audit. |
| v2 junction candidate construction | Implemented, with several corrections | Balanced re-pairing cycles, coordinate/orientation constraints, missing-coordinate handling and mandatory masking tested. |
| Initial v2 training | One-epoch execution smoke completed | No full 30-model v2 campaign or established v2 biological superiority. The smoke predates the latest filter correction. |
| Full native candidate audit | First attempt failed correctly | One-hop node filtering removed individual candidates and broke endpoint balance. Failure evidence preserved. |
| Latest v2 filtering fix | Implemented and tested; pushed as `28728fd` | Excludes whole affected masking groups; reports loss; preserves v1 filtering and the canonical graph. Corrected server audit not yet run. |
| Native geometry/degree baselines | Implemented and tested | Fixed linear controls fit training chromosomes and score validation chromosomes only; no held-out chromosome scores. Full real-data output pending. |
| Larger, sequence-conditioned or path-aware variants | Not established improvements | Existing code/proposals do not count as trained, compared and validated models. Each requires separate attribution and controls. |

The first 20-window diagnostic still showed a distance cue. Removing an endpoint
imbalance fixes a correctness problem; it does not prove that all shortcuts are
gone or that biological performance improved.

## 5. Incomplete, blocked, or optional work

| Item | Blocker / remaining requirement | Suggested priority |
|---|---|---|
| Full H/random inference and corrected v2 audit | Fresh server access and completed prediction checks | Immediate |
| Stronger sequence comparison | Fixed local sequence context, pinned model/input/pooling, same examples and unchanged splits | Next major scientific control |
| Full v2 pretraining and frozen transfer | Candidate/masking readiness, declared validation-only model-selection protocol, matched comparators | After readiness and control review |
| HG008 external SV validation | 24/30 passed; six fail the unchanged historical-probe gate. Original fitted probes/embedding snapshots were not retained; numerical diagnostics do not restore them. | Resolve provenance or explicitly redesign a new prospective external experiment; do not loosen the existing gate |
| Path auxiliary task | Canonical SV graph lacks compatible paths; full-resolution GBZ IDs do not directly match canonical segments | Conditional research track |
| Strict unseen-donor/population transfer | Donor-excluded graph and audited sample/ancestry metadata | Required for that specific generalization claim |
| Haplotype ASE and methylation | Matched donor labels and compatible haplotype embeddings | Conditional on resources; RNA locus prediction does not complete this task |
| Genotypability / broader SV types | Independent measured genotyping-accuracy labels; current inversion VCF supplies GT, not accuracy | Feasibility incomplete; no result claimed |
| GTEx, 3D contacts, molecular QTLs, variant effects | Suitable matched labels; compact GTEx matching currently too small (92/34/42 loci) | Defer until core controls resolve |
| Population-diversity scaling | Sample-diversity design and compatible donor-excluded resources | Distinct from completed window scaling |
| Final manuscript administration | Author-provided funding, contributions and acknowledgements | Author input |

## 6. Recommended roadmap and decision points

### Step 1 — Close the experiments already launched

- [ ] Authenticate, collect current server receipts, inspect failures and disk usage.
- [ ] Finish/audit all 120 H/random evaluations and all 90 new EN-TEx jobs.
- [ ] Update one consolidated result table with all outcomes, including nulls.

**Decision:** does T beat R and add value beyond H consistently? If not, revise
the learned-representation claim and focus on the objective/input representation;
adding more benchmarks would not settle this question.

### Step 2 — Make the v2 objective valid before scaling it

- [x] Correct whole-group filtering and test native loading/masking.
- [ ] Deploy the fix and audit all 1,216 canonical manifest windows.
- [ ] Quantify retained/excluded groups by chromosome/context and fit the
  validation-only geometry/visible-degree controls.
- [ ] Review remaining cues and lock the training protocol before biological
  model selection. Retain failed attempts and reasons for changes.

**Decision:** proceed only with usable coverage and a defensible task. Passing
software checks or a weak linear control alone is insufficient.

### Step 3 — Test a controlled v2 improvement

- [ ] Complete a corrected smoke/pilot, then the declared fold/seed/context run.
- [ ] Compare frozen v2 against frozen v1, H, R and C+S on identical examples.
- [ ] Change one major factor at a time: objective, graph context, capacity or
  sequence-conditioned inputs. Keep topology-native T distinct from sequence T.
- [ ] Save checkpoints, exact embeddings, fitted probes and predictions so future
  external evaluation does not require reconstructing historical models.

**Decision:** promote based on reproducible comparative evidence, not a single
favorable assay/fold or a larger reconstruction score.

### Step 4 — Strengthen the generalization evidence

- [ ] Run a task-faithful local/stronger sequence comparator on common examples.
- [ ] Establish a genuinely independent external evaluation for the chosen model.
- [ ] Start path/donor-specific extensions only after resource compatibility is
  demonstrated. Keep these claims separate from chromosome-held-out transfer.

### Step 5 — Finalize the manuscript around the evidence

- [x] Revise scientific wording, labels, donor aliases, intervals and model details;
  retain null findings and disclose intrinsic-benchmark limitations.
- [x] Build and visually check the working manuscript revision.
- [ ] Integrate the final H/random matrix and corrected v2 results when available.
- [ ] Finalize author fields and a complete reproducibility package.

For Tuesday's meeting, lead with completed EN-TEx results, the repeated-measurement
sensitivity, and the distinction between useful graph features and learned
pretraining value. Treat new assays as exploratory and v2 as work in progress.

## 7. Where the work lives

| Artifact | Location |
|---|---|
| Detailed checked list | [FOUNDATION_CAMPAIGN_CHECKLIST.md](FOUNDATION_CAMPAIGN_CHECKLIST.md) |
| Team-facing EN-TEx results and figures | [ENTEX_LAB_MEETING_20260929.md](ENTEX_LAB_MEETING_20260929.md) |
| Structural/scaling results | [FOUNDATION_CAMPAIGN_20260924.md](FOUNDATION_CAMPAIGN_20260924.md) |
| Independent v2 review | [V2_EVIDENCE_REVIEW_20260927.md](V2_EVIDENCE_REVIEW_20260927.md) |
| Fixed new panel and native audit commands | [FOUNDATION_EVIDENCE_EXTENSION_20260927.md](FOUNDATION_EVIDENCE_EXTENSION_20260927.md) |
| Working manuscript | [main.tex](../manuscript/revision_20260924/main.tex) |
| Recent review branch | `codex/v2-evidence-review-20260927` |
| Other LLM's branch | `v2/shortcut-free-pretraining-20260926` — preserved, not reset |

Server execution is isolated in `PangenomeFM_evidence_20260927` (H/random jobs),
`PangenomeFM_evidence_report_20260927` (new panel/reporting), and
`PangenomeFM_readiness_20260927` (native v2 audit). Active jobs remain pinned to
their launch code. The latest filtering correction is on GitHub, but its server
deployment failed when SSH expired; do not assume every checkout is at the
same revision.

Verification at the latest code revision: **300 tests passed**, scoped Ruff and
compile checks passed, and the original four cached manuscript gains were
unchanged. This verifies implementation and recorded arithmetic, not success of
the outstanding scientific hypotheses.

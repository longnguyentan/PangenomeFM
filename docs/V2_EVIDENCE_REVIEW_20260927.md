# PangenomeFM: independent evidence review and v2 analysis plan

Verified 27 September 2026 UTC (26 September EDT). Review branch:
`codex/v2-evidence-review-20260927`, based on the other LLM's `b39e317`.

## Takeaway

Frozen transfer is real but task dependent; v2 needs stronger controls and a corrected candidate protocol before promotion.

The completed EN-TEx and scaling experiments provide measurable, reproducible
transfer results, including null and negative findings. Equal-locus weighting
weakens the CTCF result, while H3K27ac retains a small positive effect. RNA
allelic-imbalance prediction is inconclusive across five chromosome folds.
The original reconstruction benchmark has a masking-related shortcut, so its
high scores alone cannot establish learned biological structure. The reviewed
v2 sampler removes specific endpoint and coordinate cues, but distance remains
predictive and independent controls are still necessary. Neither “best model”
nor journal readiness follows from implementing the proposed components.

## 1. Evidence and execution status

This is a custom research pipeline, not a registered NGS Workbench workflow.
The Workbench registry contained no runs and only a local compute target; its
analysis-design/results skills were used to review the existing evidence.
Authorized shell SSH accessed `tuv43532@cis-chen`; no new graph was downloaded.
The supplied review and professor-goal summary are attributed discussion inputs,
not independent meeting minutes or executable instructions.

| Evidence | Status | Scope and conclusion |
|---|---|---|
| Original cCRE/SV regression | observed, cached check | Original four gains remain +.004008/+.003406 and +.034246/+.040090; not retraining |
| Primary EN-TEx | observed, complete | 330 fits, modest SNV effects; P0/P1 inconclusive |
| RNA EN-TEx | observed, complete | 30 fits; strict ΔAP +.001568 [−.000075,+.004401]; one-hop +.001465 [−.000218,+.004303] |
| Measurement/donor/tissue follow-ups | observed, complete | 60 prediction runs replayed; all unweighted controls agree within 1e-12 |
| Biological scaling | observed, complete | 360 evaluations; exact C/K/S/C+S prediction invariance across fractions; task/context-dependent changes |
| Canonical HPRC masking diagnostic | observed, complete bounded subset | See below; not the full benchmark or full batched training replay |
| H/R controls | observed, complete exploratory matrix | Frozen original trained versus random encoders, H features, fold A/42, strict and one-hop, SV and cCRE; no final multi-fold claim |
| v2 training | functional smoke only | One epoch on label-blind selected windows; no completed 30-model v2 campaign or frozen v2 biological comparison |
| Broader model superiority | unknown | Stronger sequence comparators, multi-seed H/R controls and locked v2 evaluation still required |

Completed team results: [EN-TEx brief](ENTEX_LAB_MEETING_20260929.md),
[scaling and structural report](FOUNDATION_CAMPAIGN_20260924.md).
The [checked list](FOUNDATION_CAMPAIGN_CHECKLIST.md) retains outstanding items.

## 2. What the review changed in code

1. **Handcrafted H:** the declared capped-eight distance searched only three
   hops. A multi-source breadth-first traversal now computes distances 0–8,
   including disconnected nodes; a regression test checks hops 4–7 explicitly.
2. **Junction endpoints:** the proposed greedy negatives reused in-ends and
   included positives without partners. Minimum-cost permutation cycles now
   preserve each retained out-/in-end count exactly. Fixed-point dummy choices
   allow an unmatchable junction to remain unused without discarding an entire
   feasible span. Positives from unused junctions are not emitted or query-masked.
3. **Coordinate semantics:** groups preserve ordered SN coordinate systems and
   orientation pairs. Same-system distance tolerances are enforced. Cross-SN
   matching compares offsets within each corresponding system, never subtracts
   offsets belonging to distinct references. Unknown coordinate metadata are
   counted and excluded; missing SN is no longer interpreted as shared geometry.
4. **Reproducibility:** canonical pair order is invariant to reverse-equivalent
   link storage. The junction objective requires query masking. Junction
   preparation/frozen extraction no longer requires successful v1 negatives.
5. **Audit honesty:** raw and reversed heuristic AP are retained. The maximum of
   them uses evaluation labels and is labeled an optimistic diagnostic, not a
   validated model. v2 masking hides all validation/test positives and the
   current training group; it excludes stochastic drop-edge and larger packed
   groups. v1 single-query masking is explicitly separate from batched training.

All v1 defaults stay unchanged. The latest full local suite passes: **300 tests**;
scoped Ruff/compile checks also pass. The old benchmark and result files remain
unchanged. Existing helpers with ambiguous hosts/reset behavior were not used;
reviewed commands run in new server worktrees/output roots.

### Real candidate audit and its limits

The bounded audit uses the first 20 sorted strict HPRC slice paths. This selection
is not chromosome-representative and cannot estimate genome-wide retention.
A full-span permutation initially retained only 48 of 7,689 in-scope positives.
The saved failed design is `candidate_audit_strict/`. Balanced cycles retain
1,782 positives and 1,782 negatives (23.18%) in 282 groups on the same slices.
This change was made for feasibility, without optimizing model performance.

For the 676 internal test candidates of the cycle sampler, raw degree-deficit
AUROC is .503996 and degree-sum AUROC .491728. Distance remains predictive:
the evaluation-selected direction has AP .746650 at prevalence .5. Thus the
sampler suppresses the measured degree cue but is **not established as shortcut
free**. The v1 degree-matched diagnostic has deficit AUROC .999469 and reversed
preferential-attachment AP .982883. These bounded diagnostics do not by themselves
show how much the trained encoder depends on the cue or invalidate all downstream
scores. Learned-weight versus random/H controls answer a different question.

Artifacts: `results/v2_review_20260927/{candidate_audit_strict,cycle_audit_strict}/`.
Tests verify balanced marginals, coordinate signatures, tolerance, feasible partial
cycles, missing coordinates, and storage invariance. Marginal balance does not
prove absence of pairwise shortcuts or graph-construction bias.

### Exploratory frozen-versus-random controls

The bounded fold-A/seed-42 matrix completed on the authenticated Temple
checkout. It uses the unchanged chromosome split, frozen encoders and the
existing probe; only the random checkpoint replaces the trained PangenomeFM
weights. The table reports average precision and has no confidence interval
because it is one development fold. `H` is the handcrafted topology control;
`R` is the random encoder, even where the legacy metric file calls its column
`T`.

| Task | Context | Model | AP(C+S) | AP(C+S+H) | AP(C+S+T/R) | AP(C+S+H+T/R) | ΔT/R given C+S+H |
|---|---|---|---:|---:|---:|---:|---:|
| cCRE | strict | trained | 0.919607 | 0.921372 | 0.924667 | 0.925214 | +0.003842 |
| cCRE | strict | random R | 0.919607 | 0.921372 | 0.924702 | 0.925289 | +0.003917 |
| cCRE | one-hop | trained | 0.919610 | 0.921371 | 0.923538 | 0.923984 | +0.002613 |
| cCRE | one-hop | random R | 0.919610 | 0.921371 | 0.920959 | 0.921466 | +0.000095 |
| SV | strict | trained | 0.882468 | 0.915534 | 0.915307 | 0.930292 | +0.014758 |
| SV | strict | random R | 0.882468 | 0.915534 | 0.908063 | 0.928123 | +0.012589 |
| SV | one-hop | trained | 0.882468 | 0.915534 | 0.921802 | 0.925034 | +0.009500 |
| SV | one-hop | random R | 0.882468 | 0.915534 | 0.919066 | 0.926540 | +0.011006 |

These results are useful controls, not a new headline benchmark. The trained
and random encoders are nearly tied for strict cCRE, while the trained encoder
has a larger one-hop cCRE gain and a larger strict SV gain. The one-hop SV
trained-versus-random ordering reverses for the H-adjusted contrast. Because
this is a single fold with a reused development partition, it does not establish
that PangenomeFM is superior to a random encoder genome-wide. The compact
inputs and derived paired table are in
`results/v2_review_20260927/control_summary_complete/`; the exact command plan
and graph/checkpoint hashes are in its `status.json`.

## 3. Analysis plan: how to choose a better model defensibly

### Objective and scientific model

Decision: whether new graph pretraining adds reusable information over coordinates,
sequence, cheap graph statistics and an untrained graph encoder. The experimental
unit is a genomic locus or SV; donor/tissue repeats are correlated observations,
not independent replicates. Chromosome groups define partitions. The canonical
HPRC R2 graph and GRCh38 annotation versions remain fixed; graph membership is
transductive unless a donor is removed during graph construction.

### Comparisons and endpoints

- Keep the original linear probe, training/validation-only tuning, and frozen
  encoders. Primary ΔT = AP(C+S+T)−AP(C+S); add AP(C+S+H+T)−AP(C+S+H) and
  AP(C+S+T)−AP(C+S+R). R is a fresh seeded encoder with the same architecture,
  inputs, extraction rules and downstream partitions. Its output column called
  T by the reused runner must be interpreted as **R** in summaries.
- Use three independent initialization seeds for a final random-control matrix;
  do not reuse random seed 7 for all replicates. The current exploratory control
  matches the requested seed 42.
- Report all seven original feature sets plus H controls, AUROC, AP, class
  prevalence, normalized AP and paired fold/seed uncertainty. Do not call
  normalized AP fully prevalence invariant. Retain lower-performing conditions.
- For v2 reconstruction, add a geometry-only scorer and visible-degree-only
  scorer fitted on training/validation only, plus actual native masked encoder
  evaluation. Never choose score direction using final test labels.
- Separate topology-native T from sequence-conditioned `T_seq`: k-mer/NT node
  inputs introduce sequence into the representation and require new feature
  accounting. Larger capacity and bidirectional message passing are separate
  controlled ablations, not simultaneous unexplained upgrades.

### Selection and validity gates

The attached proposal uses fold-A downstream performance to promote v2. Fold A
therefore becomes development evidence. Do not subsequently present it as an
untouched test or combine its optimized score into a confirmatory five-fold CI.
The original five folds have already been inspected for v1 and this history must
be disclosed. Before further v2 biological scores, lock architecture/objective
using pretraining validation or a separately declared development partition;
keep final v2 comparisons separate from model selection. A prospective external
set is needed for a genuinely untouched confirmation after extensive reuse.

Candidate readiness requires retained counts and exclusions by chromosome,
context, internal split and coordinate system; complete group masking; balanced
labels/endpoints after all node filters; no missing required chromosome; and
sufficient events for the stated evaluation. Selection rules must be saved before
scores. No tolerance or bin is relaxed because a biological result is unfavorable.
The 24-window smoke failed when candidate filtering removed chr6; the 72-window
retry expands three fixed genomic quartiles per chromosome without changing the
held-out set. A one-epoch smoke establishes executability only.

Any H/R inference requires identical example/label hashes, feature coverage,
calibration protocol and probes. Final pretrained comparisons must distinguish
fixed architecture/epoch budgets from matched total optimization compute. A
window learning curve is not sample-diversity scaling.

## 4. Sequence and new downstream priorities

### Sequence baseline

The live Sequence Viewer verified four actual pre-tokenization inputs: lengths
6,000, 5, 6 and 6,000. Their source lengths were 10,616, 5, 6 and 8,474; the two
long examples use the manuscript's prefix/N/suffix sampling. This is an
illustrative label-blind sample, not a genome-wide length distribution or MSA.
The receipt records source/output hashes. The exported sequence is not a
contiguous GRCh38 window; no genomic feature track was incorrectly overlaid.

Next comparison should use a fixed, locus-centred GRCh38 window for CTCF/H3K27ac
and separate ref/alt representations where the endpoint is allele effect.
Retain the original segment S alongside it to distinguish **input context** from
**model capacity**. Then evaluate a frozen larger NT model and one independently
trained family on exactly the same examples, pinned revisions and pooling.
Do not select a model/context only because it wins a test set. The supplied
NT50M remains the manuscript comparator until this versioned evaluation runs.

### SV types and genotypability

The existing GRCh38 HGSVC inversion VCF contains **300 INV records**, 298 on
canonical chromosomes and two on alternate/unplaced contigs. All FILTER values
are `.` and all sample FORMAT fields are GT; this source does not provide a
short-read genotyping-accuracy label. Its source SHA is retained in
`sv_label_feasibility.json`. No INV/DUP result is claimed from the INS/DEL task.
A prospective INV-versus-other-SV task needs exact assembly event identity,
coordinate/endpoint mapping, duplicate handling and sample counts by fold.
DUP/complex events require a separate compatible truth source; do not relabel
insertions as duplications from sequence length alone.

Genotypability should use independent leave-one-out genotype concordance or
validated per-variant error rates, with read depth, allele frequency, size and
repeat/complexity strata. PASS/non-PASS would learn a caller's filter policy;
it is not automatically biological genotypability. HGSVC3's PanGenie bubble
callset differs from its PAV event callset and often uses CHM13 coordinates;
identifier/build joins must be proven, not guessed. Keep the canonical graph.

### Papers informing these decisions

- [FakeEdge, Dong et al. 2022](https://proceedings.mlr.press/v198/dong22a.html)
  identifies target-edge-induced connectivity shift in link prediction. It
  motivates protocol-matched masking controls; it does not prove this project's
  exact shortcut or establish junction re-pairing as novel.
- [Macias-Velasco et al. 2026](https://www.nature.com/articles/s41467-026-73663-3)
  benchmarks reference choice across functional assays. Global effects are
  modest with important locus-specific differences, supporting local/allelic
  questions while warning that representation and analysis pipelines interact.
- [Logsdon et al. 2025](https://www.nature.com/articles/s41586-025-09140-6)
  provides 65 assembled genomes and PanGenie evaluation, including leave-one-out
  concordance. This motivates measured genotyping outcomes rather than graph-
  derived SV-presence labels or an arbitrary FILTER surrogate.
- [Benchmarking DNA foundation models, 2025](https://www.nature.com/articles/s41467-025-65823-8)
  evaluates frozen embeddings across tasks and emphasizes input compatibility.
  It supports matching context and endpoints before claiming a stronger model.
- [Nucleotide Transformer, 2025](https://www.nature.com/articles/s41592-024-02523-z)
  provides a natural within-family capacity comparison. A larger encoder alone
  is not evidence of a better pangenome model.

## 5. Professor goals, manuscript claims and execution

| Supplied goal | Concrete response |
|---|---|
| Methods-first contribution | State the masking problem and controlled objective change; establish H/R/context controls before claims of novelty or superiority |
| Broader biology / EN-TEx | Complete primary + sensitivities + RNA + weighting analyses; report all nulls in the team brief |
| Complex regions / SV genotyping | Preserve unfavorable complexity interactions; use independently measured genotype accuracy for the next task |
| Other pretrained models | Match examples, biological input window, pooling and frozen-probe capacity; keep original NT50M benchmark |
| More samples / consortium harmonization | Current scaling varies windows only; donor-excluded graph and ancestry metadata are still required |
| NMI / other venue | No empirical gate guarantees acceptance. Decide after mechanism, utility and fair-comparator evidence, not a single favorable fold |
| Manuscript administration / release | Funding IDs, authors, contributions and permissions require author facts; never invent them |

Reproduce controls from the review worktree, with a fresh output directory:

```bash
PYTHONPATH=src:. python scripts/server/run_v2_review_controls.py \
  --main-checkout /home/tuv43532/PangenomeFM \
  --out-root results/v2_review_20260927/controls --fold fold_a --seed 42
PYTHONPATH=src:. python scripts/audit_masking_degree_shortcut.py \
  --slices '/home/tuv43532/PangenomeFM/server_workspace/data/benchmarks/hprc_r2_pretrain_5mb_paired/*_strict_segments.csv.gz' \
  --max-slices 20 --out-dir results/v2_review_20260927/cycle_audit_strict
PYTHONPATH=src:. python -m pytest -q
```

The runner refuses an existing root, verifies the canonical graph checksum,
requires one exact checkpoint, records all argv and commit IDs, and stops on a
failed command. Controls run at commit `972d0de`; the separate sampler audit/smoke
worktree runs `6337685`. Later documentation synchronization does not change
those recorded provenance identities. No authentication is currently needed.

## Update: native one-hop filtering defect (27 September)

The full-manifest native audit found a correctness issue beyond the bounded
strict-slice sampler audit: canonical oriented handles may be absent from the
stored-direction node set, and rowwise filtering can break a group's balanced
endpoint counts. The v2 loader now excludes complete affected groups and records
the loss; original v1 behavior is preserved. Tests cover mixed link storage.
The old one-epoch v2 smoke is not a validation of the corrected loader. A fresh
full-manifest audit with validation-only geometry/visible-degree controls is
required; see the [evidence extension](FOUNDATION_EVIDENCE_EXTENSION_20260927.md).

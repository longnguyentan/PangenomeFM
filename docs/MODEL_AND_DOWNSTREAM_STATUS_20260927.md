# PangenomeFM: model and downstream evidence status

Updated 27 September 2026. Technical work only; no manuscript edits.
Branch: `codex/v2-evidence-review-20260927`.

## What changed in this execution

The frozen two-branch candidate passes the prespecified **single-fold
development gate**. It preserves the SV strength of topology-only v2 and the
cCRE strength of the NT-conditioned encoder. It beats all three same-dimension
partial/random branch controls after C+S+H on both tasks. This is a candidate
worth replicating, not an established best model or an acceptance guarantee.

- [x] Complete eight new frozen biological probes: four branch combinations ×
  SV/cCRE; replay metrics and verify identical examples, labels and baselines.
- [x] Complete trained/random coordinate-stream ablations and verify the same
  2,050 reconstruction validation candidates and frozen NT cache.
- [x] Consolidate **52 context rows across 26 evaluation groups**, including
  all EN-TEx tasks, sensitivities, tissues, ENCODE subsets and external HG008.
  These are not 26 independent new benchmarks.
- [x] Map both official TraitGym matched datasets without dropping examples;
  verify all 14,780 reference alleles against the actual HPRC graph sequence.
- [x] Audit the existing HGSVC3 inversion annotations: 300 source events,
  298 mapped primary-chromosome events; preserve the two unmapped contig events.
- [x] Freeze the candidate and remaining-seed replication protocol before
  observing the new seeds.
- [ ] Finish seeds 314159 and 20260806: launched in a durable server session.
- [ ] Complete the fixed 4,000-iteration probe sensitivity: log review found
  some SV logistic fits reached the manuscript's 800-iteration cap. The original
  runs remain unchanged; candidate promotion also requires this numerical check.
- [ ] Replicate across chromosome folds and run the selected candidate on new
  downstream endpoints; the full v2 matrix is not completed.

Detailed earlier repairs and failures remain in
[the execution record](MODEL_REPAIR_EXECUTION_20260927.md).

## Complete downstream inventory and actual performance

The full, reproducible [Markdown scorecard](../results/foundation_evidence_20260927/task_scorecard/completed_downstream_tasks.md)
and [CSV](../results/foundation_evidence_20260927/task_scorecard/completed_downstream_tasks.csv)
include C+S AP, C+S+T AP, paired gain, 95% interval, run count and source path.
Every gain is checked against the difference of the corresponding means.
This is source-table consolidation, not retraining or a fresh replay of every
historical prediction. No favorable-result selection is applied.

| Family | Completed scope | What the evidence supports |
|---|---|---|
| ENCODE cCRE | Binary; PLS, pELS, dELS, CTCF-only versus common background; three complexity strata | Small original T gains. Subclass rows use subsets of binary-probe predictions, not separately trained subtype models. |
| HGSVC3 SV | INS versus DEL; length, frequency, complexity and donor strata | Strong original T contribution. This is not SV discovery or a breakpoint/non-breakpoint task. |
| EN-TEx P0 | Global AS-prone cCRE, exposure matched, H3K27ac-only, CTCF-only, complexity analyses | Global effect inconclusive; some predefined sensitivity effects have positive bootstrap intervals. Exposure matching changes the population/prevalence. |
| EN-TEx P1 | Five distal-enhancer tissue tasks and macro average | Macro effect inconclusive. Retain all tissues, including negative/inconclusive effects. |
| EN-TEx P2 / RNA | CTCF, H3K27ac, RNA ASE, ATAC, H3K4me3, H3K27me3 | CTCF/H3K27ac gains are small; H3K4me3 has positive bootstrap intervals. RNA, ATAC and H3K27me3 are inconclusive in the main paired comparisons. |
| EN-TEx follow-ups | Donor, exposure/equal-locus, complexity, normalized AP, AUROC and multiplicity checks | Completed supporting analyses; these do not create independent new benchmark replications. |
| HG008 | 30/30 prospective refit evaluations, 69 clonal INS/DEL variants | Gains +0.032525 strict / +0.064863 one-hop, both with intervals crossing zero. One genome; no HG008 training. Six historical replay failures remain distinct from successful prospective refits. |
| H/R controls | Full 120-evaluation v1 matrix | Strict SV trained-minus-random after C+S+H +0.005295; one-hop SV and cCRE provide weaker evidence. Handcrafted/random controls are essential. |
| Scaling | 120 intrinsic and 360 biological runs complete | More data helps strict SV and one-hop cCRE, but does not uniformly improve biological transfer. v1 reconstruction retains the documented shortcut limitation. |
| Graph transfer | Historical cross-resource/release evaluations | Completed under v1; corrected-objective transfer has not been established. |

The complete EN-TEx primary/sensitivity/tissue/RNA/extension matrix contains
450 runs. Completed does not mean a positive hypothesis test. Five-fold
hierarchical bootstrap intervals are also not equivalent to five-cluster
exact sign-flip significance; the latter has limited resolution.

## Frozen architecture candidate

**Candidate specification is fixed; scientific validation is incomplete.**

| Part | Fixed choice |
|---|---|
| Reference resource | Exact manuscript HPRC R2 SV graph and 5 Mb windows; unchanged checksums |
| Topology branch T | Existing dual-stream encoder; hidden 48, two layers, four heads; typed bidirectional graph messages; coordinate attention with multiscale/orientation RoPE |
| T inputs | Existing seven structural/coordinate inputs; degree recomputed on visible graph; unmasked component identity removed |
| Sequence-context branch Q | Same encoder dimensions; seven inputs plus 512D frozen NT-v2 50M segment embeddings |
| Pretraining | Degree-balanced junction re-pairing, span size 16; whole-group masking; reverse and reverse-complement hidden links removed; signed-offset/gap bins with ratio 1.25 |
| Heads | Existing MLP for T; linear products/absolute-differences interaction scorer for Q; choices selected by reconstruction validation before biological fitting |
| Representation | Ordered concatenation [T,Q], 48+48=96 dimensions; no fitted fusion weights |
| Biological learning | Existing logistic probe only; both encoders and NT remain frozen |
| Scope | One-hop development; strict remains ineligible under the current geometry-matching protocol |

No new sequence model, graph release, biological fine-tuning, tissue selection,
or learned biological fusion gate was introduced. Q is sequence-conditioned;
the combined representation must not be called topology-only T.

The [analysis plan](../configs/frozen_branch_analysis_plan_20260927.json) specifies
all controls, validity checks, decision criteria and limitations. The extractor
rejects different branch universes/counts/policies. Probes verify both
checkpoints' test and validation chromosomes, context and seed.

### Observed validation AUPRC

Fold A, seed 42, one-hop. Same 91,932 training / 43,217 validation SV examples
and 164,435 / 73,988 cCRE examples. **No test-chromosome predictions.**

| Representation E | SV C+S+E | SV C+S+H+E | cCRE C+S+E | cCRE C+S+H+E |
|---|---:|---:|---:|---:|
| T only | 0.911535 | 0.916828 | 0.916260 | 0.916574 |
| Q only | 0.892733 | 0.905693 | 0.920388 | 0.920628 |
| T + Q | **0.914380** | **0.918485** | **0.920474** | **0.920716** |
| T + random Q | 0.911685 | 0.916654 | 0.919189 | 0.919687 |
| Random T + Q | 0.912297 | 0.915482 | 0.920153 | 0.920347 |
| Both random | 0.909148 | 0.913255 | 0.919008 | 0.919493 |

After C+S+H, the two trained branches beat both-random by +0.005230 SV and
+0.001223 cCRE. Replacing only trained T with random T loses +0.003003 SV
and +0.000368 cCRE. Replacing only trained Q loses +0.001831 SV and +0.001029
cCRE. Thus the cCRE evidence for T's learned weights remains small.

The extra gain over the better single branch is +0.001657 SV and +0.000088
cCRE after C+S+H. Do not describe the tiny cCRE difference as a replicated
improvement. The practical result is preserving both task strengths with one
fixed representation while passing the same-dimension controls.

**Numerical caveat:** subsequent log inspection found two convergence warnings
in each seed-42 composite SV probe log, and none in the corresponding cCRE logs.
The historical logs did not identify which feature fit issued each warning.
The apparent development-gate pass is therefore provisional pending a fixed
4,000-iteration sensitivity with per-feature iteration/convergence receipts.
Only the optimizer's iteration ceiling changes; estimator, class weighting,
regularization, features, checkpoints and examples remain fixed. This is not
a search for a better biological score.

[Audited per-run table](../results/foundation_evidence_20260927/composite_biological_analysis/audited_per_run.csv)
· [Paired contrasts](../results/foundation_evidence_20260927/composite_biological_analysis/paired_differences.csv)
· [Gate](../results/foundation_evidence_20260927/composite_biological_analysis/development_gate.json)
· [Gain figure](../results/foundation_evidence_20260927/composite_biological_analysis/frozen_branch_gains.pdf).
All existing SV strata, including sparse/undefined bins, remain in
`all_sv_strata.csv`.

## Graph-message ablation

Same NT inputs, candidates, partitions, linear head and maximum 10-epoch budget.
The coordinate stream retains visible degree; it is **not graph-free**.

| Model | Validation AP | Window-macro AUROC |
|---|---:|---:|
| Full NT-conditioned encoder, trained | 0.869981 | 0.904021 |
| Coordinate stream only, trained | 0.730937 | 0.756674 |
| Full NT-conditioned encoder, frozen random | 0.654747 | 0.693320 |
| Coordinate stream only, frozen random | 0.611729 | 0.622205 |

The trained full model exceeds its message ablation by +0.139044 AP.
This supports the graph-message component in this reconstruction setting.
It does not prove that every shortcut is removed or that the gain generalizes
to biological labels. Trained/random weights, exact candidate identities,
cache provenance and saved checkpoint scores pass the native report checks.

[Coordinate-ablation outputs](../results/foundation_evidence_20260927/nt_coordinate_analysis/validation_metrics.csv)
and [full-model outputs](../results/foundation_evidence_20260927/nt_junction_analysis/validation_metrics.csv)
retain complete metrics and audits.

## New downstream feasibility: actual data audits

### TraitGym

Pinned official dataset revision:
`1fde19555fe8c0a55b1382bdf4c6f7082209f566`; inspected author-code commit:
`4d80fe889415ffc02d45c7cb5446a326288819bd`.

| Official dataset | Examples | Positive | Unique loci | Graph / NT coverage | REF matches |
|---|---:|---:|---:|---:|---:|
| Complex traits, matched 1:9 | 11,400 | 1,140 | 11,400 | 100% / 100% | 11,400/11,400 |
| Mendelian traits, matched 1:9 | 3,380 | 338 | 3,354 | 100% / 100% | 3,380/3,380 |

No multi-segment SNVs or mixed-label loci. All rows and supplied match groups
are preserved. Positives and negatives both have 100% coverage. Chromosome X
has 500 Mendelian examples and is already covered by the existing fold C test
partition. An initial conversational assertion that X lacked a fold was
incorrect and was corrected against the actual configuration.

Coordinates are GRCh38, 1-based POS, mapped to [POS-1,POS), with no nearest
fallback. All 14,780 REF alleles match the original graph sequence. The two
viewer examples are deterministic source-order samples, independent of labels
or model scores. [Coverage and reference audit](../results/foundation_evidence_20260927/traitgym_reference_audit/audit.json).

**Next implementation:** official leave-one-chromosome-out probing and
chromosome-weighted AP, including the provided matched controls and published
sequence-feature comparators. Existing five-fold results would be an explicitly
adapted protocol, not directly comparable leaderboard numbers. T is a locus
prior; it does not distinguish two alleles at the same locus. No TraitGym
classifier or performance result has been claimed in this audit.
[Official implementation and metric](https://github.com/songlab-cal/TraitGym).

### HGSVC3 inversion / multiclass SV

The already downloaded annotation has **300 events**, including two nonprimary
contig events. All 298 events on primary chromosomes map both first/last affected
bases and have NT features. Test-fold support is 72, 74, 59, 42 and 51 inversions.
The two nonprimary events remain explicitly recorded, not silently discarded.
The historical mapper and binary probe now reject INV/DUP/other classes and
inconsistent binary targets, preventing accidental relabeling as DEL.

Use the TSV's SV-Pop BED+6 coordinate contract, verified against each event ID
and length. Preserve INV as its own type; the historical binary INS/DEL probe
must never label inversions as deletions.
[Readiness audit](../results/foundation_evidence_20260927/inversion_readiness/audit.json)
· [Fold support](../results/foundation_evidence_20260927/inversion_readiness/fold_support.csv).

**Next implementation:** disjoint event-level INS/DEL/INV labels, overlap
handling, one-vs-rest AP and length-matched controls. Complex events are not
automatically a mutually exclusive fourth class. The paper's broader complex-SV
count is not a ready GRCh38 multiclass table.
[HGSVC3 paper](https://www.nature.com/articles/s41586-025-09140-6),
[SV-Pop format](https://github.com/EichlerLab/svpop).

## Novelty and remaining evidence, in priority order

The defensible methods direction is **audited pangenome junction
self-supervision that controls endpoint-degree and genomic-geometry shortcuts,
with frozen biological reuse**. Whole-group masking and matched re-pairing are
the research-specific parts to assess. Bidirectionality, sequence concatenation,
linear probes and masked graph pretraining are not individually new.
GraphMAE already studies masked graph-feature reconstruction and MGAE studies
masked graph autoencoding. A complete novelty claim requires direct comparisons
to these approaches, not a new model name.
[GraphMAE](https://arxiv.org/abs/2205.10803),
[MGAE](https://arxiv.org/abs/2201.02534).

| Priority | Remaining work | Present status / concrete next action |
|---|---|---|
| 1 | Seed replication of fixed T+Q | Running seeds 314159/20260806 with matched random twins; every result retained |
| 2 | Chromosome replication and v2 task transfer | Not run; freeze protocol before additional label-informed choices |
| 3 | Stronger sequence / nonlinear probe comparisons | Not completed; use identical examples and input budgets, including sequence truncation controls |
| 4 | TraitGym | Data, mapping and all REF checks pass; official-protocol classifier still needed |
| 5 | INV-containing SV type task | 298 primary events pass mapping/NT coverage; multiclass probe/overlap controls still needed |
| 6 | DART-Eval | Official suite identified; select coordinate-anchored tasks and audit coverage before claiming a comparable result |
| 7 | Measured SV genotypability | Variant-level leave-one-out outcomes and callable denominators still missing; FILTER, SVR and self-genotyping are not valid replacements |
| 8 | GTEx / SV-expression / trait links | Compact positive sets insufficient; obtain all-tested/callable universes before defining negatives |
| 9 | Path-aware RNA/methylation, donor scaling | Need verified haplotype-to-canonical-segment correspondence and donor-excluded graph design |
| 10 | Cross-graph correspondence / construction invariance | Need nontrivial sequence/assembly truth, coordinate controls and duplicate/sample-overlap audits |
| 11 | MPRA, constraint, tandem repeats, clinical SVs | Endpoint/ascertainment feasibility first; no completed PangenomeFM performance claims |
| 12 | Strict repaired pretraining | Fails matched-candidate support; do not relax matching to obtain a favorable number |

DART-Eval distinguishes zero-shot, probing and fine-tuning tracks; synthetic
sequences do not necessarily have an unambiguous genomic graph location.
[Official DART-Eval](https://github.com/kundajelab/DART-Eval).

The Genomic Intelligence catalog was inspected: available chromatin models use
supervised ENCODE labels, and its expression model uses ENCODE/GTEx-related data.
They can be declared supervised comparators after overlap review, not independent
ground truth or substitutes for the frozen NT baseline.
[Catalog snapshot](../results/foundation_evidence_20260927/plugin_model_catalog.json).
NGS Analysis Workbench's design skill guided the unit/contrast/validity-gate
plan. Biological Sequence & Alignment Viewer opens the actual reference
windows; its display is not a substitute for the complete programmatic REF check.

## Reproduction and server continuation

Run from the evidence worktree with `PYTHONPATH=src:.`; use the existing
`pangenomefm-server` environment. No passwords are stored in commands/files.

```bash
# Rebuild the 52-row historical scorecard from the bundled compact sources.
PYTHONPATH=src:. python scripts/server/build_downstream_scorecard.py \
  --out-dir <fresh-scorecard-output>

# Replay the completed composite comparison on actual predictions.
PYTHONPATH=src:. python -m tasks.transfer.frozen_branch_report \
  --root <readiness-results>/composite_biological_validation \
  --topology-root <readiness-results>/biological_validation_common \
  --sequence-root <readiness-results>/nt_biological_validation \
  --out-dir <fresh-composite-report>

# Run the unchanged two-seed replication; all commands/receipts are saved.
PYTHONPATH=src:. python scripts/server/run_frozen_branch_seed_replication.py \
  --main-checkout /home/tuv43532/PangenomeFM \
  --audit-root /home/tuv43532/PangenomeFM_model_repair_20260927/results/foundation_evidence_20260927 \
  --nt-cache <evidence-results>/benchmark_nt_completion/benchmark_nt.npz \
  --topology-control-cache <existing-H-cache.npz> \
  --decision-gate <evidence-results>/composite_biological_analysis/development_gate.json \
  --out-root <fresh-replication-output> --execute

# After both seeds finish, retain every paired contrast; no chromosome CI.
PYTHONPATH=src:. python -m tasks.transfer.frozen_branch_replication_report \
  --sources <evidence-results>/composite_biological_analysis \
    <replication-output>/seed_314159/biological_analysis \
    <replication-output>/seed_20260806/biological_analysis \
  --out-dir <fresh-seed-summary>

# Repeat the complete TraitGym mapping/reference audit on the pinned files.
PYTHONPATH=src:. python scripts/server/audit_traitgym_coverage.py \
  --dataset complex_traits=data/external/traitgym_1fde195/complex_traits_matched_9.parquet \
  --dataset mendelian_traits=data/external/traitgym_1fde195/mendelian_traits_matched_9.parquet \
  --feature-cache completed_nt=<evidence-results>/benchmark_nt_completion/benchmark_nt.npz \
  --full-segments /home/tuv43532/PangenomeFM/server_workspace/data/processed/hprc_r2_sv/full_segments.csv.gz \
  --verify-reference --out-dir <fresh-traitgym-audit>

# Audit the already downloaded inversion annotation; retain unmapped events.
PYTHONPATH=src:. python scripts/server/audit_inversion_readiness.py \
  --annotation /home/tuv43532/PangenomeFM/server_workspace/data/downstream/sv/hgsvc3/1.0/GRCh38/variants_GRCh38_sv_inv_HGSVC2024v1.0.tsv.gz \
  --full-segments /home/tuv43532/PangenomeFM/server_workspace/data/processed/hprc_r2_sv/full_segments.csv.gz \
  --sequence-cache <evidence-results>/benchmark_nt_completion/benchmark_nt.npz \
  --out-dir <fresh-inversion-audit>
```

Actual active server root:
`/home/tuv43532/PangenomeFM_evidence_report_20260927/results/foundation_evidence_20260927/frozen_branch_seed_replication`.
Tmux socket `evidence-20260927`, session `frozen_branch_seed_replication`.
`status.json` records stage completion and failures; each child retains commands,
native checkpoints, predictions and logs. A launched stage is not a completed
result. Historical and other-LLM worktrees are preserved.

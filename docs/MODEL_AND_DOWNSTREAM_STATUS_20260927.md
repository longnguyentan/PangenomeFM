# PangenomeFM: model and downstream evidence status

Updated 27 September 2026. Technical work only; no manuscript edits.
Branch: `codex/v2-evidence-review-20260927`.

## What changed in this execution

The frozen two-branch candidate passed the prespecified **seed-42 development
gate**. The first additional seed reproduces its SV advantage, but the cCRE
comparison against a random topology branch is slightly negative. Therefore
the trained-topology benefit for cCRE is not consistently replicated. The
remaining seed and fixed solver-convergence sensitivity are running. The
candidate specification is fixed; a final superiority claim is not supported.

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
- [x] Finish seed 314159: 4 pretraining fits and 16 biological probes; all
  native audits completed, including the unfavorable cCRE contrast.
- [ ] Finish seed 20260806: running in the same durable server session.
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
The [complete gain overview](../results/foundation_evidence_20260927/task_scorecard/downstream_topology_gains.pdf)
shows every context row and interval; external HG008 is displayed separately
because its interval is much wider. Panel scales are explicitly marked.

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

### Corrections to the supplied external summaries

The attached suggestions contain useful directions and several stale or
incorrect details. Actual code/data/results take precedence:

- The original server resources are accessible and the EN-TEx matrix is
  complete; this is no longer a missing-graph/data-preparation-only project.
- The historical SV endpoint is INS versus DEL, not breakpoint detection.
  An inversion row cannot be added by changing its binary label to DEL.
- Strict/one-hop describes graph context, not a TSS classification rule.
- CTCF/H3K27ac SNV experiments use the full accessible EN-TEx call set;
  the downloaded high-confidence file contains RNA-seq measurements.
- Window-count scaling has completed; it is not donor/haplotype-count scaling.
- HG008 prospective probe refits completed, while exact historical replay is
  a different claim. Positive point estimates with wide intervals crossing
  zero are inconclusive, not an automatic scientific pass.
- A shared segment vector is a locus representation, not an allele-specific
  embedding. No arbitrary haplotype ID or mismatched GBZ node frequency was
  added to make an allele-aware claim.
- The repository uses audited NPZ caches and exact SHA-256 resource identities;
  illustrative CSV filenames or commands in the suggestions are not real
  replacements for the repository's pipeline.

No supplied document instruction to change the graph, fine-tune NT with labels,
or substitute a success threshold overrides the frozen, controlled evaluation.

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
regularization, feature definitions, checkpoints and examples remain fixed.
The probes re-extract frozen features on GPU, which can introduce small
floating-point differences; every score change cannot be attributed solely
to the larger iteration budget. This is not
a search for a better biological score.

[Audited per-run table](../results/foundation_evidence_20260927/composite_biological_analysis/audited_per_run.csv)
· [Paired contrasts](../results/foundation_evidence_20260927/composite_biological_analysis/paired_differences.csv)
· [Gate](../results/foundation_evidence_20260927/composite_biological_analysis/development_gate.json)
· [Gain figure](../results/foundation_evidence_20260927/composite_biological_analysis/frozen_branch_gains.pdf).
All existing SV strata, including sparse/undefined bins, remain in
`all_sv_strata.csv`.

### First additional initialization seed: 314159

All sixteen biological probes completed using the fixed protocol, with the
same validation examples and invariant C+S/C+S+H predictions across models.
These remain development-validation scores, not independent chromosome tests.

| Task, after C+S+H | T+Q AP | Both-random AP | Gain over both-random | Gain over random T+Q | Gain over T+random Q |
|---|---:|---:|---:|---:|---:|
| SV INS/DEL | 0.917347 | 0.914144 | +0.003203 | +0.002027 | +0.000754 |
| cCRE | 0.920702 | 0.919643 | +0.001059 | **-0.000096** | +0.001212 |

The inherited per-seed gate reports **not_promoted** because cCRE fails the
trained-topology-versus-random-topology comparison. This result is retained;
no replacement seed, head, feature choice or threshold was selected. The final
seed continues unchanged. The numerical solver sensitivity is separate.

Reconstruction gains replicate qualitatively: topology trained/random AP is
0.731156/0.596269; NT-conditioned trained/random AP is 0.863401/0.626892.
All use the same 2,050 candidates, exact initialization controls and fixed
sequence cache. Better reconstruction still does not guarantee every
biological contrast is positive.

[Seed 314159 biological table](../results/foundation_evidence_20260927/frozen_branch_seed_replication/seed_314159/biological_analysis/audited_per_run.csv)
· [All paired contrasts](../results/foundation_evidence_20260927/frozen_branch_seed_replication/seed_314159/biological_analysis/paired_differences.csv)
· [Recorded failed gate](../results/foundation_evidence_20260927/frozen_branch_seed_replication/seed_314159/biological_analysis/development_gate.json).

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

A further relevant baseline is **Topology Only Pre-Training (ToP)**, published
in May 2026. It removes node/edge attributes during contrastive pretraining and
studies transfer across graph domains. Thus topology-only pretraining and
multi-domain graph transfer are already established directions. Its reported
fine-tuning protocol differs from our frozen probes; an adapted comparison
would need to state that difference. It strengthens the case for evaluating
our junction objective against established graph self-supervision, rather than
claiming novelty from topology inputs alone.
[Davies et al., 2026](https://link.springer.com/article/10.1007/s10618-026-01210-1).

| Priority | Remaining work | Present status / concrete next action |
|---|---|---|
| 1 | Seed replication and numerical convergence | Seed 314159 complete, mixed cCRE evidence; seed 20260806 biological probes and fixed solver sensitivity running |
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

An additional public genotypability search inspected all **22,627 entries** in
the HGSVC3 publication working-data manifest. No filenames matched leave-one-out,
PanGenie, concordance or genotyping terms; the official README describes assembly
QC/annotation products. This bounds the search, not proof that outcomes do not
exist elsewhere or inside differently named files. The required per-variant
outcomes remain unresolved.
[Inventory audit](../results/foundation_evidence_20260927/genotypability_public_inventory.json)
· [Official archive README](https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/HGSVC3/working/20241218_phase3-main-pub_data/20241218_phase3-main-pub_data.README.txt).

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
The `frozen_branch_seed_report` tmux session waits for that existing experiment
and automatically creates the complete three-seed report at
`<readiness-results>/frozen_branch_seed_replication_analysis`. It records an
explicit failure if a source job fails or an eight-hour wait expires, and does
not treat a failed scientific gate as a missing result.
[Exact finalizer](../results/foundation_evidence_20260927/finalize_seed_replication.py).

The separate convergence experiment runs in
`/home/tuv43532/PangenomeFM_readiness_20260927/results/foundation_evidence_20260927/probe_convergence_4000`,
tmux session `probe_convergence_4000` on the same socket. Its
[exact command receipt](../results/foundation_evidence_20260927/probe_convergence_4000_launch.json)
and [launcher](../results/foundation_evidence_20260927/run_probe_convergence_4000.sh)
retain commit `c86d447`, all original checkpoint paths, and the fixed optimizer
budget. The same sensitivity is being extended to seeds 314159 and 20260806
because the first replication also has SV convergence warnings. All eight
model arms are included for both tasks, irrespective of performance.
The bounded continuation reuses the original recorded commands and changes
only the output root and uniform iteration ceiling; original results remain
intact. A convergence-aware three-seed report retains numerical failures as
well as failed performance gates.
[Continuation code](../scripts/server/run_frozen_branch_convergence.py). The native report will mark `optimization_incomplete` if any compared
probe remains unconverged. Use a fresh output root to reproduce it.

Validation: **372 local tests passed**, including bitwise default-probe
prediction equivalence; the nine targeted label/convergence tests also pass
in the server environment. The label-integrity check accepts all 173,969
historical INS/DEL examples (110,623 INS; 63,346 DEL). Cached manuscript AP
regression checks pass; they are not retraining.

# Review of the proposed model and downstream roadmap

Reviewed 27 September 2026, 19:47 UTC. This is an evidence review and proposed
decision protocol, not a report of new model training. The supplied review is a
discussion input; its commands, deadlines and journal recommendations are not
treated as verified project state.

**Recommendation:** adopt the priority on learned-weight attribution, a valid v2
objective and a small controlled model comparison. Finish the existing EN-TEx
panel, then pause further assay expansion. Treat richer inputs and path-aware
modelling as separate, measured extensions. Neither a positive single-fold
result nor the removal of one shortcut establishes foundation-model superiority.

## 1. What the random-encoder result actually says

The archived fold-A/seed-42 controls support concern about the contribution of
pretraining. They do not show equivalence across tasks or contexts. The replayed
AP increments below use identical feature-set contrasts within each task:

| Task/context | Trained T added to C+S+H | Random R added to C+S+H | Trained minus random, conditional on C+S+H |
|---|---:|---:|---:|
| SV strict | +0.014758 | +0.012589 | +0.002169 |
| SV one-hop | +0.009500 | +0.011006 | −0.001506 |
| cCRE strict | +0.003842 | +0.003917 | −0.000075 |
| cCRE one-hop | +0.002613 | +0.000095 | +0.002518 |

Source: [audited paired differences](../results/foundation_evidence_20260927/hr_original_audit/paired_gains.csv).
These are one-fold observations without inferential intervals. In strict SV,
trained versus random differs by +0.007243 without H and +0.002169 with H;
the conditioning set matters. The table is the original exploratory matrix,
not the pending full-control matrix.

A random GNN is a useful nonlinear transformation of real node features and
graph connectivity. Its success does not mean that the graph has no useful
information. It means that the added value of **learned weights** remains
unestablished. Conversely, beating R alone would not show that T beats cheap H.
The proposed full 120 evaluations are the appropriate next attribution check.
They are correlated fold/seed/task evaluations, not 120 independent replicates.

Sun et al. provide relevant motivation for careful controls, but their molecular
graph findings do not diagnose the cause in our pangenome experiment.
[Primary paper](https://arxiv.org/abs/2207.06010).

## 2. Corrections to the proposed diagnosis

| Suggestion | Assessment and implication |
|---|---|
| The graph is too small | Plausible hypothesis, not demonstrated cause. Exact segment/edge counts should come from the canonical graph receipt, including orientation and edge-count conventions. More correlated nodes or graphs do not automatically add independent information. |
| Strict windows have no alternative alleles | Too absolute. Strict slicing selects reference-anchored core segments and retains induced links. It excludes off-reference sequence branches, but can retain variant-related bypass connections. Measure branch/bubble coverage before changing the window definition. |
| Messages only flow in one direction | True for the graph aggregation branch along stored directed edges. The dual-stream model also attends to coordinate neighbours in both directions; the whole encoder is not blind to all successors. |
| Seven simple node features limit learning | Seven base features are confirmed: offset, length, rank, reference status, degree, orientation and component ID. Their simplicity does not prove they cause the random/trained tie. Some are geometric rather than purely topological. |
| Haplotype frequencies give information a random encoder cannot invent | Frequencies add information to the input, but R can exploit that same input too. Compare raw frequency features, H, R and T with identical frequency inputs. An input gain must not be called a pretraining gain. |
| A weak pair scorer guarantees better embeddings | No guarantee. Even a linear scorer can exploit geometry or candidate-construction cues. Compare geometry-only, frozen/random-embedding and learned-embedding scorers under the same masking protocol. |
| A shared SNV vector is a limitation | Correct for allele-specific effect prediction. Current static T supports locus susceptibility prediction; it cannot distinguish two alleles at the same locus by itself. This does not invalidate the explicitly locus-level EN-TEx tasks. |

Code inspected: [strict/one-hop slicing](../src/graph/slicing.py),
[base features and graph aggregation](../src/models/gat.py),
[dual-stream coordinate attention](../src/models/dual_stream_gat.py).

## 3. The v2 objective needs the latest audit before another large run

The supplied review predates several corrections. We fixed negative endpoint
reuse, coordinate/orientation grouping, mandatory query masking and then a
native-loader defect: filtering individual candidates with unavailable nodes
destroyed otherwise balanced endpoint counts. Commit `28728fd` filters entire
affected groups, counts the loss and preserves v1 behaviour.

The first native audit failed on real one-hop data; that failure is retained.
The corrected full audit is **not yet run**. The latest local suite previously
passed 300 tests, but passing tests is not a real-data readiness result. A bounded
earlier audit still showed a distance cue. Use “junction re-pairing with audited
endpoint balance,” not an unconditional “shortcut-free objective.”

Next steps, in order:

- [x] Correct native whole-group filtering and add regression coverage.
- [ ] Deploy `28728fd`; audit all 1,216 canonical manifest windows.
- [ ] Inspect retained groups by chromosome/context, including excluded groups.
- [ ] Run the declared training/validation-only degree and geometry controls.
- [ ] Verify masking after packing, node filtering, reverse-equivalent edge
  handling and DropEdge; do not rely only on the standalone sampler.
- [ ] Lock a small v2 training and frozen-transfer comparison after readiness.

See [execution and failure evidence](FOUNDATION_EVIDENCE_EXTENSION_20260927.md).

## 4. A smaller, interpretable model programme

### First: objective and directionality

Keep canonical v1 as the frozen reference. Train the corrected v2 with current
inputs first, then add bidirectional graph messages as one separate variant.
An ordinary reverse message `(v,u)` differs from the reverse-complement graph
edge `(v^1,u^1)`. Remove every representation of a hidden junction, and declare
which edges count toward structural input features. Match architecture, inputs,
dimension and probe tuning when constructing each random twin.

Use fixed-capacity decoder ablations. A larger pair-geometry scorer might raise
reconstruction while contributing little reusable embedding information. Masked
feature reconstruction is a reasonable later auxiliary objective, provided the
target is genuinely hidden from all inputs. GraphMAE motivates testing this
approach; it does not establish that reconstructing these seven particular
features will improve biological transfer.
[GraphMAE](https://arxiv.org/abs/2205.10803).

### Second: richer context and inputs, separately

1. **Bubble-aware context:** define and audit complete retained alternatives,
   finite size limits, truncation and coverage. Keep target loci, folds and
   downstream units fixed. More context is a distinct experimental factor.
2. **Path support:** first demonstrate a path-to-canonical-segment correspondence.
   Record unique traversing haplotypes and the eligible/callable denominator,
   rather than treating repeated traversals as additional individuals. The
   canonical SV graph currently lacks compatible path annotations.
3. **Sequence-conditioned graph encoder:** retain frozen NT and the original
   topology-only T as distinct model variants. The manuscript verifies 303,425
   required segments and complete original-task coverage; this does not prove
   complete coverage of all oriented nodes in newly expanded windows. Audit
   missingness and strand handling. End-sampling affects 26.3% of the required
   segments, so a stronger task-local sequence comparator is also valuable.
4. **Multiple graphs:** start with one verified HPRC/HGSVC correspondence pilot,
   then consider another resource. GRCh38 position alone is not a unique allele
   identity. Handle CHM13 coordinates, paralogs, one-to-many matches and donor
   overlap explicitly; withhold homologous loci across resources together.

For every added input X, evaluate direct X and matched random/learned encoders
receiving X. A raw-feature or no-message control and a modest nonlinear probe
can distinguish input/feature-map benefits from message-passing and pretraining
benefits. Use the same examples and validation budget throughout.

## 5. Resource claims: promising, with specific boundaries

- **HPRC2:** the July 2026 preprint reports 460 haplotypes. This is not
  interchangeable with the 454 phased paths in our audited subset or the 462
  haplotypes in an older configuration. Pin the exact new graph and sample
  manifest in a separate v3 resource version; preserve the manuscript graph.
  [HPRC2 preprint](https://www.biorxiv.org/content/10.64898/2026.07.21.739710v1).
- **Matched functional data:** the Kinnex repository documents 206 samples,
  LCL-derived RNA, alignments and expression matrices. These are valuable
  starting points, but gene/transcript abundance is not automatically a phased
  ASE label. Verify donor/path identity, phasing, read assignment, coverage and
  mapping bias. A full graph has sequence segments and paths; a node is not
  necessarily one complete biological allele.
  [Kinnex analysis and data](https://github.com/wwliao/hprc_release2_kinnex_analysis).
- **Epigenome:** the official registry provides methylation, expression and
  accessibility resources. It does not establish that every assay has the same
  donors, tissues, processing or callable regions. Build the sample intersection
  before promising a joint prediction task.
  [HPRC epigenome registry](https://registry.opendata.aws/hprc-epigenome/).
- **Expanded SV callset:** 614,522 refers to the combined 1,218-sample CHM13
  INS/DEL callset; its GRCh38 counterpart contains 587,779. The 293 assembly
  genomes are a component, not the entire source. These data overlap HPRC/HGSVC
  and do not automatically constitute independent external validation or supply
  new inversion/complex-type labels.
  [UCSC resource documentation](https://hgdownload.soe.ucsc.edu/gbdb/hs1/hubs/public/lrSv1kLin.html),
  [author release](https://github.com/jiadong324/1KG_LongRead_SV).

These resources make a path-aware research programme plausible. They do not
automatically remove our canonical-node correspondence, matched-label or
unseen-donor evaluation blockers. A one-to-three-week completion estimate is
not yet supported by a resource/compute audit.

## 6. Downstream priorities after the control gate

| Task | Recommendation | Required validity check |
|---|---|---|
| SV INS/DEL | Keep as a principal existing task | Use the correct task name and quantify benefit beyond H/R; do not call it SV discovery. |
| INV/complex SV types | First new label-feasibility audit | Count mapped examples by chromosome/type. Group every component/breakend of an event together. Check nested/overlapping type definitions before fitting multiclass probes; include length, sequence and graph baselines. |
| Measured genotypability | Highest-value new application if labels exist | Obtain per-locus or per-allele leave-one-out correctness with callable denominators, including failures. Predicting concordance does not itself improve a genotyper. FILTER/GT alone are insufficient. |
| TraitGym | Good first standard-benchmark feasibility audit | Retain its official backgrounds, chromosome evaluation and metric; audit graph coverage. Static T can be a locus prior in a frozen probe, not an allele-effect scorer. |
| DART-Eval | Select compatible coordinate-anchored tasks | Keep zero-shot, probed and fine-tuned tracks distinct. Unmapped synthetic sequences and counterfactual allele changes need a different representation; disclose any restricted subset. |
| SV–expression / trait association | Conditional next-stage task | Require the tested variant–gene/trait universe and measurement power. Do not label every unreported pair negative. |
| Cross-graph retrieval | Useful harmonisation diagnostic | Ground truth needs sequence/assembly correspondence. Compare coordinate, sequence and H baselines; prevent supplied coordinates from trivially revealing the match. Not automatically cheap or biological validation. |
| Constraint, repeat variability, MPRA, clinical SVs | Defer until a specific unit/label/baseline is ready | Audit ascertainment, population overlap, sequence/length confounds and whether the target merely restates a supplied input. |

HGSVC3 reports 298 GRCh38 inversions and 1,852 resolved complex SVs, but these
headline counts are not a ready disjoint classification table. Its detailed
complex-event analyses use additional definitions and reference coordinates.
It also reports measured PanGenie and Locityper leave-one-out experiments; we
still need usable per-locus outputs, including unsuccessful targets.
[HGSVC3 primary paper](https://www.nature.com/articles/s41586-025-09140-6).

The new SV resource reports 890 novel SV-associated eQTLs and 105 significantly
associated SVs, not a complete set of positive and negative variant–trait pairs.
Association and causal effect are different endpoints.
[Lin et al. preprint](https://www.medrxiv.org/content/10.64898/2026.08.21.26361050v1.full).

TraitGym supports chromosome-based logistic probes and a chromosome-weighted AP
metric; DART-Eval distinguishes zero-shot, probing and fine-tuning settings.
These designs provide useful comparisons when preserved, rather than replacing
their candidate sets with a convenient mapped subset without reporting losses.
[TraitGym implementation](https://github.com/songlab-cal/TraitGym),
[DART-Eval](https://arxiv.org/abs/2412.05430).

## 7. Replace the one-fold promotion/venue rule

Fold A is already inspected. It can reject a broken implementation or screen
compute feasibility; it cannot provide an untouched confirmatory test. Likewise,
“within 0.005” is a proposed tolerance, not evidence of noninferiority without an
uncertainty analysis and a scientific justification.

For the full control report, preserve all task/context cells and show:

- `AP(C+S+H+T) − AP(C+S+H)`;
- `AP(C+S+T) − AP(C+S+R)`;
- `AP(C+S+H+T) − AP(C+S+H+R)`;
- paired fold/seed estimates, fold-clustered uncertainty, raw AP/AUROC,
  prevalence, all folds and multiplicity sensitivity.

Keep training/validation model selection separate from outer test reporting.
All original v1 test folds have already been inspected, so subsequent model
development is exploratory on those benchmarks. Fix the small variant set in
advance, report failed variants and seek genuinely independent confirmation.

**Decision:** if T adds reproducible benefit beyond both H and R, deepen that
finding. If R/H explain the gain, describe graph-derived predictive information
and revise the learned-pretraining claim. If estimates are uncertain, retain the
uncertainty. No outcome guarantees NMI or Nature Methods acceptance; neither
journal should be selected by one favourable fold. The proposed subsection claim
about neighbourhood context must also await supporting controls.

## 8. Correct the status before the lab meeting

- [x] Core EN-TEx: 330 runs completed.
- [x] RNA: 30 additional runs completed; both global gain intervals cross zero.
- [x] Biological scaling: all 360 evaluations completed; transfer is not
  universally monotonic. This is separate from 360 total core-plus-RNA EN-TEx jobs.
- [x] Equal-locus sensitivity: CTCF becomes inconclusive; H3K27ac stays positive.
- [ ] Full H/R controls: last observed 96/120; final completion unverified.
- [ ] Fixed ATAC/H3K4me3/H3K27me3 panel: last observed 2/90; collect all outcomes.
- [ ] Corrected native v2 audit: awaiting server authentication and deployment.
- [ ] HG008: six replay-gate failures remain; no full external result claimed.
- [ ] Additional tasks above: proposals/feasibility work, not completed results.

Do not equate “mostly null” with “nothing learned”: exposure and locus-weighting
sensitivities materially limit which biological claims are defensible. Complete
the declared panel without selecting assays by gain, then pause expansion.

The next server action is to refresh receipts and run the corrected audit using
the established Temple connection. The pasted helper's conflicting host is not
verified and was not executed. SSH authentication is currently unavailable;
this review did not change server jobs or produce new experiment results.

Current overview: [project status and roadmap](PROJECT_STATUS_AND_ROADMAP_20260927.md).
Team results: [EN-TEx lab-meeting brief](ENTEX_LAB_MEETING_20260929.md).

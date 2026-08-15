# PangenomeFM next-stage decision audit — 2026-08-15

This document supersedes `NEXT_STAGE_PHASE0_AUDIT_20260814.md`. The earlier
audit was correct when written, but several server matrices and downstream
artifacts that were then missing have since been completed, packaged, verified,
and imported locally.

## Current evidence state

The verified execution package contains complete ablation, link-baseline,
cCRE, structural-variant, capacity-scaling, path-materialization, and matched
enrichment outputs. The only submission-tier experiment that remains blocked is
the shared-donor-excluded transfer experiment. The available HPRC SV-scale GFA
does not retain embedded sample paths, so producing a donor-filtered graph from
that file would not be scientifically valid.

The present work therefore does not launch another broad GPU matrix. It targets
the two issues that most directly control the interpretation of the existing
results:

1. candidate identity must be invariant to reverse-complement traversal; and
2. PangenomeFM must be compared against the strongest simple topology
   baselines within frozen graph-complexity and context strata.

## Why this is now the primary analysis

The overall baseline table is not consistent with a universal claim of graph
pretraining superiority. In strict/core slices, preferential attachment
outperforms PangenomeFM in pooled AUPRC. In endpoint-expanded slices,
PangenomeFM substantially outperforms the same topology heuristic. The useful
scientific question is consequently not “does PangenomeFM always win?” It is:

> Under which graph-complexity and context-exposure conditions does pretrained
> representation learning add value beyond simple graph topology?

The new analysis makes this comparison on exact common candidates and reports
PangenomeFM-minus-baseline AUPRC, AUROC, and negative-log-likelihood differences
by low/medium/high frozen complexity, context, fold, and chromosome. Confidence
intervals resample chromosomes; candidate edges are never treated as
independent inferential units.

## Correctness changes

Candidate identity is now defined as

```text
(u, v) == (v xor 1, u xor 1)
```

for oriented node identifiers. Positive sets, random negatives, coordinate-hard
negatives, distance-matched negatives, and paired distance-matched negatives
all enforce this identity. Generated negatives are unique under canonical
identity and cannot be the reverse traversal of a positive. Candidate creation
hard-fails if any canonical identity is duplicated or assigned conflicting
labels.

The trainer already removes both the exact positive query traversal and its
reverse complement from message passing. It also rejects conflicting candidate
labels before splitting. Random initialization, NumPy, Python, CUDA, and
DropEdge are seeded for new runs. Historical runs predate part of this contract;
the manuscript must distinguish historical execution from the corrected code
path rather than retroactively describing old runs as if they used the fix.

## Complexity-v2 definition

The strict/core graph determines the locus category. The score is an
equal-weight mean of robustly normalized, winsorized features:

- log edges per kilobase;
- branching-node fraction;
- log maximum degree;
- log cycle rank per kilobase;
- log nodes per kilobase; and
- log robust node-length heterogeneity.

Medians, median absolute deviations, and tertile thresholds are frozen from the
strict reference population before model or baseline performance is loaded.
Expanded-context node and edge exposure ratios are descriptive covariates, not
components of the frozen locus score.

## Legacy expanded candidates

The historical native 5-Mb expanded predictions contain a very small number of
reverse-equivalent duplicate/conflicting identities. New generation forbids
them. For analysis of the already completed matrix, strict context uses a hard
error policy. Expanded context uses an explicit legacy-exclusion policy:
every representation of a conflicting canonical identity is excluded from both
methods, same-label reverse representations are averaged into one biological
candidate, and all removals are written to the candidate-coverage audit.

This is preferable to silently selecting a label or counting the same
biological relation twice. A future full rerun should instead use regenerated
canonical candidates.

## Local validation completed

The canonical 50-kb benchmark was regenerated as
`benchmark_matched_nonoverlap_v3_canonical_20260815`.

- retained matched slices: 450;
- candidate rows: 68,896;
- positive rows: 34,448;
- negative rows: 34,448;
- exact duplicate rows: 0;
- reverse-equivalent duplicate rows: 0;
- orientation-equivalent label conflicts: 0; and
- invalid labels: 0.

Fifteen attempted loci lacked a feasible matched expanded negative and were
excluded from both contexts. This leaves 225 matched loci rather than pretending
that all 240 attempted loci are comparable.

Complexity-v2 passed on all 450 retained slice/context rows. The strict
population contains 225 loci and freezes exactly 75 low-, 75 medium-, and 75
high-complexity loci. No performance columns were read during fitting.

## Comparator decision

DeepGene is an approximate downstream representation comparator, not an exact
masked-edge comparator: it serializes graph-derived sequence units and does not
provide native candidate-edge scoring. PangenomeX is a shallow-WGS CNV
classification pipeline with BAM/VCF/read-depth inputs and a different target.
It remains contextual unless a separate same-sample CNV study is designed.

Neither method should be inserted into a single ranked masked-link-prediction
table by changing its task until the names look comparable.

## Decision gates after the server run

The next manuscript revision may claim context- and complexity-dependent added
value only if all of the following hold:

1. strict complexity audit status is `PASS`;
2. all eight dense-score runs complete with exact common-slice coverage;
3. every legacy expanded exclusion is counted and preserved;
4. chromosome-block intervals, not edge-level intervals, are reported;
5. preferential attachment and degree sum are included as primary baselines;
6. strict and expanded contexts remain separate tasks; and
7. null or negative strata are reported without selection.

Section 11 remains blocked and is not a prerequisite for this analysis.

# Complete graph contexts and attention repair — 28 September 2026

## Outcome

All **608 original one-hop intervals** now have separately materialized graph
contexts that preserve their original nodes/edges and include every complete
alternative component attached to the original reference core, plus all of that
component's reference anchors. The 751,237-segment HPRC R2 graph is unchanged.
No biological labels, replacement graph, new haplotype assumptions, or learned
weights were used to construct these contexts.

This removes a concrete limitation in the training inputs: **605/608 original
one-hop windows contained incomplete alternative components**. It is a completed
data/engineering result. Wider-context biological performance has not been fitted.

## Completed QC

| Check | Result |
|---|---:|
| Original intervals / completed / failed | 608 / 608 / 0 |
| Unique represented segments | 742,607 |
| Unique alternative segments | 439,182 |
| Alternative components represented | 135,043 of 135,902 |
| Whole-graph frozen NT coverage in new contexts | 100% |
| Original one-hop reference flanks removed | 0 |
| Windows with isolated core nodes | 0 |
| Cross-reference-contig links in original graph | 0 |
| Train–validation, train–test, validation–test shared segment IDs | 0 in each of five folds |
| Median / maximum segments per context | 638 / 28,229 |
| Largest native oriented-handle count | 28,287 |

The remaining 859 alternative components are outside the original primary-
chromosome window universe; the earlier whole-graph audit identifies these as
nonprimary-reference components. They were not discarded because of their
size, labels, or performance. New anchors do not recursively recruit additional
components. Components are defined by undirected connectivity after removing
reference nodes; they are **not validated directed bubbles or phased alleles**.
Stored GFA link orientation is preserved.

[All window counts and memory bounds](../results/foundation_evidence_20260928/component_contexts/windows.csv)
· [Five actual-ID split audits](../results/foundation_evidence_20260928/component_contexts/fold_overlap.csv)
· [Source checksums and preparation audit](../results/foundation_evidence_20260928/component_contexts/audit.json).

## Attention issue and explicit compatibility policy

The historical local coordinate-attention branch computes `q[source]*k[destination]`
but normalizes and aggregates into `destination` using `v[source]`. This differs
from query-normalized attention. Clamped neighbor ranks can also repeat boundary
keys. The regression test demonstrates both issues on a small graph; it does not
infer any biological performance change.

Two implementations are available explicitly:

- `--coordinate_attention_mode chunked_exact`: global scaled dot-product
  attention, processed in query chunks, with recomputation during backward.
  This preserves the mathematical global attention rule. Floating-point and
  dropout ordering can differ from the historical implementation.
- `--coordinate_attention_mode chunked_window --window_k 128`: corrected
  query-normalized local attention, unique in-range keys, stable position ties,
  and self-attention. The radius is `floor(window_k/2)`. This is an explicit
  change to historical sparse semantics and requires a separate experiment.

`--attention_chunk_size 512` controls peak chunk size. Both options preserve
architecture parameter shapes and save their mode in checkpoint arguments.
Old checkpoints and all current experiments retain `legacy` by default.
The current masked-feature chromosome study uses **dense global attention**;
this sparse-branch issue does not change its execution or interpretation.

An independent comparison with the pre-change Git module verifies **bitwise
identical legacy outputs and gradients**, both train/eval and dense/sparse.
[Compatibility receipt](../results/foundation_evidence_20260928/attention_compatibility/legacy_bitwise_replay.json).
The new modes match independently computed dense equations and gradients;
checkpoint recomputation preserves dropout for a fixed chunked run, and saved
model metadata restores the selected implementation exactly.

The implementation follows PyTorch's documented
[scaled dot-product attention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
and [activation-checkpointing](https://docs.pytorch.org/docs/2.14/checkpoint.html)
contracts. This is a correctness and memory repair, not a novel attention method.

## Largest-context execution

A real forward/backward pass was run on chr1:0–5,000,000, using the existing
48D, two-layer, four-head encoder with 519 input features, graph messages,
segment-grouped masking, and the shallow masked-feature decoder. No optimizer
ran, no weights changed, and no biological outcome was read. Zero/unit target
moments were used solely for profiling, not fitted as a new training protocol.

| CPU resource smoke | Exact global chunks | Corrected local chunks |
|---|---:|---:|
| Native oriented nodes | 28,287 | 28,287 |
| Forward + backward time, two CPU threads | 212.89 s | 9.79 s |
| Whole-process peak RSS, including graph/cache loading | 4.94 GiB | 8.18 GiB |
| Finite loss and gradients | Pass | Pass |
| Encoder weights unchanged | Pass | Pass |

Timing is a single resource check, not a rigorous speed benchmark. CPU peak RSS
is not CUDA memory. The unchunked score tensor alone would occupy 12.80 GB
(11.92 GiB) **per attention layer**, before softmax, backward, other activations
and optimizer state. No complex window was omitted to fit memory. GPU profiling
is a separate queued check after the existing biological experiment and original-
model reference release the GPUs; wider-context pretraining is not yet complete.

## Reproduce on the dedicated server checkout

```bash
cd /home/tuv43532/PangenomeFM_evidence_report_20260927
export PYTHONPATH=src:.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=/home/tuv43532/miniconda3/envs/pangenomefm-server/bin/python

# Requires a fresh output root. Never overwrite the original benchmark.
"$PY" scripts/server/prepare_component_contexts.py \
  --config configs/component_contexts_20260928.json \
  --out-dir results/component_contexts_reproduction --materialize

"$PY" scripts/server/verify_component_contexts.py \
  --contexts results/component_contexts_reproduction \
  --out results/component_contexts_reproduction/materialization_replay.json

"$PY" scripts/server/profile_component_attention.py \
  --contexts results/component_contexts_reproduction \
  --checkpoint results/foundation_evidence_20260927/masked_feature_full/seed_42/full_trained/checkpoint.pt \
  --mode chunked_exact --device cpu \
  --out-dir results/component_context_exact_cpu_reproduction
# Repeat with --mode chunked_window and a different output root.
```

Native completed graph tables and membership NPZ stay on the server under
`results/foundation_evidence_20260927/component_contexts_20260928/`.
Only compact audits/tables are versioned. The completed [608-window replay](../results/foundation_evidence_20260928/component_contexts/materialization_replay.json) verifies every materialized
segment row and every induced oriented edge against the original graph before
future training; changing an orientation is covered by a rejection test.

## Verification status

Full local suite: **456 passed**, 14 known warnings. Server targeted tests:
**25 passed**. New-module Ruff, compilation and diff checks pass. All 608 saved
contexts independently match original segment metadata and oriented induced
edges. An intentional changed-orientation test is rejected. Neither synthetic
tests nor resource-profile losses are reported as biological performance.

## Next scientific decision

Finish the already fixed trained/random chromosome comparison and original-v1
reference. The new input preparation and optional attention modes do not change
those protocols or select from partial test scores. A wider-context experiment
must fix its training population, target weighting, contexts, controls and compute
budget separately. Better alternative-node coverage and a successful memory smoke
are prerequisites, not proof of a biological improvement or a finalized model.

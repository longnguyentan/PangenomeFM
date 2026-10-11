# PangenomeFM v3 research prototype

Sequence-aware multiallelic imputation over a haplotype-by-site matrix.
This is an independently implemented PyTorch research prototype. Model accuracy,
external validation and biological improvement have **not** been established.

## Implemented components

* Complete-coverage DNA CNN with explicit A/C/G/T/N and padding. Overlapping
  windows require four-base halos and disjoint core masks; coverage must equal
  the full allele length. It is not a reproduction of a pretrained DNA model.
* Chunked linkage attention, local convolution, and population attention.
  Reference rows attend only to reference rows. Query rows read the reference
  panel, and never update it, preventing query-to-panel-to-query leakage.
* Explicit missing (-1) and masked (-2) tokens. Per-site allele indices are local
  namespaces. Donor IDs, variant IDs and population frequencies are not embedded.
* A flat, ragged complete-candidate output head. Exact checkpointed blocked
  log-sum-exp retains all candidates, including sites with 65,535 candidates.
  It avoids padding every site to the largest candidate count.
* Separate random/span masking and SNP-only query imputation masks. Nested and
  overlapping target closure is explicit and must be applied by the data loader.
* Sample identity and REF/ALT effect-orientation checks for prospective QTL/SMR
  analyses. These checks do not perform QTL discovery or establish causality.

The default backbone has width176, eight attention heads and six axial layers.
Its actual GPU feasibility must be profiled on the intended complete panel,
window and candidate shapes before training. There are no released trained
weights or benchmark performance claims in this branch.

## Minimal use

Python3.11 and PyTorch2.6 are the development environment. Add this directory to
PYTHONPATH, then import ModelConfig from axial_model and RaggedPangenomeImputer
from ragged_model_v1. The input schema is provided in input_schema.json.
Call hidden_ragged(observations, allele_vectors, offsets, site_features,
panel_rows), then ragged_nll(hidden, allele_vectors, offsets, truth, scored).
The loss is a sum; its returned count is the number of scored observations.
Missing truths must not be included in scored. Keep all missing/unsupported
samples and loci in separately reported denominators.

For a defensible experiment, freeze family splits, masks, complete source
identities, references, candidate sets, optimizer, budgets and endpoints before
execution. Compare a paired frozen-random backbone, TRAIN allele frequencies,
nearest copying, and matched established imputers. Hash full source files before
opening and after EOF; timestamps alone do not authenticate content. Keep DEV
and external query rows out of the reference panel, and keep sealed tests closed.
Software conformance is not evidence of biological performance.

## Attribution and scientific boundaries

The chunking design is inspired by [STICI](https://github.com/shilab/STICI)
([paper](https://www.nature.com/articles/s41467-025-56273-3)). Upstream STICI
already supports multiallelic output and uses TensorFlow. This implementation
adds axial population attention and sequence-aware candidate representations;
it is not an upstream model/weight port. No STICI source or weights are bundled.

Independent external evaluation still requires donor/relative overlap checks,
assembly and allele harmonization, and truth QC. QTL discovery counts alone
require calibration and replication before a biological improvement claim.
MAGE's [current source](https://github.com/mccoy-lab/MAGE) describes a corrected
sample/library swap; resolve processed sample identity before using expression
matrices. SMR additionally needs harmonized effects, matched LD and HEIDI, and
does not by itself demonstrate causality.

This source-only branch contains no cohort data, private plans, transcripts,
server configuration, historical checkpoints or private research Git history.

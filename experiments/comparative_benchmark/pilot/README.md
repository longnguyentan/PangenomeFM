# Comparative benchmark pilot

This directory contains the version-controlled pilot specification. Generated data products are written to `results/comparative_benchmark/pilot/` so source and outputs remain separate.

Pilot scope:

- local HPRC matched 50 kb benchmark (segment/link inventory exactly matches the verified HPRC R2 inventory; raw-source checksum provenance still must be copied before publication);
- chr22 (`fold_b` test chromosome);
- strict/core and endpoint-expanded contexts treated as separate tasks;
- split seed `20260806` and model seeds `42`, `314159`, `20260806`;
- exact PangenomeFM input mapping;
- approximately matched DeepGene endpoint-sequence inventory;
- contextual-only PangenomeX status.

Passing the pilot means that the manifest, graph integrity, labels, folds and per-method conversion audit pass. It does **not** mean that three exact comparable model scores exist. `--require-exact-all` intentionally exits nonzero while DeepGene and PangenomeX remain task-incompatible.

Current gate: the chr22 example manifest passes after 127 repeated source rows are removed, but the full 480-slice source audit finds 28 orientation-equivalent label conflicts in 12 slices outside the selected pilot chromosome. Full fold-B training is blocked until those candidates are regenerated. A deterministic one-epoch chr22-only inner-split smoke passed in about 14.2 seconds at about 1.82 GiB peak RAM; its score is not held-out evidence.

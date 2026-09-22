"""Prespecified falsification controls using the unchanged manuscript classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.server.run_ccre_frozen_probe_fold import evaluate_feature_sets


def chromosome_permutation(chromosomes: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = np.arange(len(chromosomes))
    for chrom in sorted(set(chromosomes)):
        positions = np.flatnonzero(chromosomes == chrom)
        order[positions] = rng.permutation(positions)
    return order


def evaluate_controls(
    *,
    selected: pd.DataFrame,
    components: dict,
    test_chrs: set,
    val_chrs: set,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    chrom = selected.chrom.to_numpy()
    y = selected.label.to_numpy().copy()
    c, s, t = (
        components[k]
        for k in ["coordinate", "frozen_sequence_fm", "frozen_pangenomefm"]
    )
    order = chromosome_permutation(chrom, seed + 271828)
    shuffled_t = np.concatenate([c, s, t[order]], axis=1)
    train = ~np.isin(chrom, sorted(test_chrs | val_chrs))
    y[train] = y[order[train]]  # order never crosses chromosome or partition.
    outputs = []
    for name, matrix, labels in [
        ("CST_shuffled_T", shuffled_t, selected.label.to_numpy()),
        ("CST_shuffled_training_labels", np.concatenate([c, s, t], axis=1), y),
    ]:
        metrics, _, predictions = evaluate_feature_sets(
            segids=selected.example_id.to_numpy(),
            chromosomes=chrom,
            labels=labels,
            features={name: matrix},
            test_chrs=test_chrs,
            val_chrs=val_chrs,
            seed=seed,
            feature_access={name: "prespecified within-chromosome negative control"},
        )
        outputs.append((metrics, predictions))
    return pd.concat([x[0] for x in outputs]), pd.concat([x[1] for x in outputs])

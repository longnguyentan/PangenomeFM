from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tasks.ccre.embedding_baseline import _manifest_target_chromosome


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "server"
    / "run_ccre_frozen_probe_fold.py"
)
SPEC = importlib.util.spec_from_file_location("ccre_frozen_probe", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_manifest_chromosome_aliases() -> None:
    assert _manifest_target_chromosome("GRCh38#0#chr22") == "chr22"
    assert _manifest_target_chromosome("id=CHM13|chrX") == "chrX"
    assert _manifest_target_chromosome("22") == "chr22"


def test_checkpoint_holdout_is_verified(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    torch.save(
        {
            "args": {"test_chrs": ["chr1", "chr6"], "val_chrs": ["chr2"], "seed": 42},
            "closure": "strict",
        },
        checkpoint,
    )
    audit = MODULE.validate_checkpoint_holdout(
        checkpoint,
        test_chrs={"1", "chr6"},
        closure="strict",
        seed=42,
    )
    assert audit["checkpoint_test_chromosomes"] == ["chr1", "chr6"]


def test_all_features_use_identical_test_nodes_and_validation_calibration() -> None:
    chromosomes = np.repeat(["chr1", "chr2", "chr3", "chr4", "chr5", "chr6"], 8)
    labels = np.tile([0, 1], len(chromosomes) // 2).astype(np.int8)
    signal = labels[:, None].astype(np.float32) + np.linspace(0, 0.1, len(labels))[:, None]
    features = {
        "coordinate": signal,
        "graph": signal,
        "structural": signal,
        "sequence_kmer": signal,
        "frozen_pangenomefm": signal,
        "sequence_plus_frozen_pangenomefm": np.concatenate([signal, signal], axis=1),
        "training_prevalence": np.empty((len(labels), 0)),
        "random_uniform": np.empty((len(labels), 0)),
    }
    metrics, per_chromosome, predictions = MODULE.evaluate_feature_sets(
        segids=np.arange(len(labels)),
        chromosomes=chromosomes,
        labels=labels,
        features=features,
        test_chrs={"chr1"},
        val_chrs={"chr2"},
        seed=42,
    )
    assert set(metrics["feature_set"]) == set(features)
    assert metrics["n_test"].nunique() == 1
    assert metrics["n_test"].iloc[0] == 8
    counts = predictions.groupby("feature_set")["segid"].apply(set)
    assert all(value == set(range(8)) for value in counts)
    assert set(per_chromosome["chromosome"]) == {"chr1"}

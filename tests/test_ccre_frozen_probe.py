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


def test_validation_development_never_scores_heldout_features_or_labels() -> None:
    import pytest
    chromosomes = np.repeat(['chr1', 'chr2', 'chr3'], 20)
    labels = np.tile([0, 1], 30).astype(np.int8)
    signal = labels[:, None].astype(np.float32)
    signal[:20] = np.nan  # Any attempted held-out predict_proba would fail.
    kwargs = dict(segids=np.arange(60), chromosomes=chromosomes, labels=labels,
                  features={'coordinate': signal}, test_chrs={'chr1'}, val_chrs={'chr2'},
                  seed=42, validation_only=True)
    metrics, chrom_metrics, predictions = MODULE.evaluate_feature_sets(**kwargs)
    assert metrics.n_test.eq(0).all()
    assert set(metrics.scope) == {'development_validation'}
    assert set(predictions.chromosome) == set(chrom_metrics.chromosome) == {'chr2'}
    changed = labels.copy()
    changed[:20] ^= 1
    other, _, other_predictions = MODULE.evaluate_feature_sets(**dict(kwargs, labels=changed))
    pd.testing.assert_frame_equal(predictions, other_predictions)
    pd.testing.assert_frame_equal(metrics, other)
    with pytest.raises(ValueError, match='overlap'):
        MODULE.evaluate_feature_sets(**dict(kwargs, val_chrs={'chr1'}))


def test_manuscript_extraction_policy_matches_window_universe_without_changing_features(tmp_path):
    from models.dual_stream_gat import DualStreamPangenomeGAT
    from tasks.ccre.embedding_baseline import _extract_embeddings
    from scripts.server.prepare_node_sequence_fm_cache import sha256_file

    segments, rows = [], []
    for window in range(2):
        ids = np.arange(30) + window * 30
        seg = pd.DataFrame(dict(name=ids.astype(str), id=ids, seq="ACGTACGTAC", LN=10, SO=ids * 10, SN="GRCh38#0#chr2", SR=0))
        links = pd.DataFrame(dict(from_seg=ids[:-1], to_seg=ids[1:], from_orient="+", to_orient="+"))
        edges = pd.DataFrame(dict(u_oid=np.r_[ids[:12] * 2, ids[:12] * 2],
                                  v_oid=np.r_[ids[1:13] * 2, ids[2:14] * 2], label=[1] * 12 + [0] * 12))
        if window == 1:
            edges = edges.iloc[:0]  # Native v2 extraction keeps this; v1 excludes it.
        paths = [tmp_path / f"{window}_{kind}.csv" for kind in ["segments", "links", "edges"]]
        for frame, path in zip([seg, links, edges], paths):
            frame.to_csv(path, index=False)
        rows.append(dict(name=f"window{window}", closure="1hop", target_sn="GRCh38#0#chr2",
                         segments_path=str(paths[0]), links_path=str(paths[1]), edge_pred_path=str(paths[2])))
        segments.append(seg)
    full, manifest = tmp_path / "full.csv", tmp_path / "manifest.csv"
    pd.concat(segments).to_csv(full, index=False)
    pd.DataFrame(rows).to_csv(manifest, index=False)
    args = dict(hidden_dim=8, n_heads=2, n_layers=1, dropout=0, stream_mode="graph",
                node_structure_source="visible", objective="junction_repair", seed=42, split_seed=20260806)
    model = DualStreamPangenomeGAT(in_dim=7, hidden_dim=8, n_heads=2, n_layers=1, dropout=0, stream_mode="graph", edge_mlp_dim=16)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(dict(model_state=model.state_dict(), args=args, in_dim=7, edge_feat_dim=0), checkpoint)
    before = sha256_file(checkpoint)
    kwargs = dict(checkpoint=checkpoint, manifest=manifest, full_segments=full,
                  labeled_segids=set(range(60)), closure="1hop", device_name="cpu", seed=42,
                  return_canonical_audit=True)
    native, _, _ = _extract_embeddings(**kwargs)
    common, _, audit = _extract_embeddings(**kwargs, extraction_candidate_policy="manuscript")
    assert set(native) == set(range(60)) and set(common) == set(range(30))
    assert audit["retained_slices"] == ["window0"]
    assert audit["checkpoint_objective"] == "junction_repair"
    for key in common:
        np.testing.assert_array_equal(common[key], native[key])
    assert sha256_file(checkpoint) == before

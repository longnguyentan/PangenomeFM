from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from tasks.transfer.junction_readiness import (
    feature_rows, fit_validation_baselines, validate_groups,
)


def native_fixture():
    u = np.array([0, 4, 0, 4, 8, 12, 8, 12, 1, 5, 1, 5])
    v = np.array([2, 6, 6, 2, 10, 14, 14, 10, 3, 7, 7, 3])
    labels = np.tile([1, 1, 0, 0], 3)
    src = np.r_[u[labels == 1], 0]
    dst = np.r_[v[labels == 1], 8]
    degree = np.bincount(src, minlength=16) + np.bincount(dst, minlength=16)
    norm = np.log1p(degree).max()
    x = np.zeros((16, 7), dtype=np.float32)
    x[:, 4] = np.log1p(degree) / norm
    return dict(name="window", target_sn="GRCh38#0#chr1", closure="strict",
                nodes=np.arange(16), node_feats=x, so_arr=np.arange(16), src=src, dst=dst,
                temps=np.ones(16), labels=labels, query_u=u, query_v=v,
                train_idx=np.arange(4), val_idx=np.arange(4, 8), test_idx=np.arange(8, 12),
                orient_arr=np.arange(16) % 2, pop_ids_arr=np.zeros(16), branching_frac=.1,
                deg_norm=norm, train_groups=[np.arange(4)], group_ids=np.repeat([0, 1, 2], 4),
                pair_geom=np.zeros((12, 4)), n_pos=6, n_neg=6)


def test_native_packed_mask_hides_all_query_and_heldout_positives():
    sd = native_fixture()
    args = Namespace(seed=42, batch_size=4, orientation_rope=False, pop_cond=False,
                     drop_edge=False, drop_edge_rate=.1)
    train = feature_rows(sd, args, "train")
    # Only structural 0 -> 8 remains after all three candidate groups are hidden.
    unit = np.log(2) / sd["deg_norm"]
    np.testing.assert_allclose(train.degree_u, [unit, 0, unit, 0])
    np.testing.assert_allclose(train.degree_v, 0)
    validation = feature_rows(sd, args, "validation")
    np.testing.assert_allclose(validation.degree_u, [unit, 0, unit, 0])
    np.testing.assert_allclose(validation.degree_v, 0)
    with pytest.raises(ValueError, match="Held-out"):
        feature_rows(sd, args, "test")
    sd["query_u"][2] = 15
    with pytest.raises(ValueError, match="Endpoint imbalance"):
        validate_groups(sd)


def test_validation_baselines_fit_only_training_labels():
    from tasks.transfer.junction_readiness import GEOMETRY, DEGREE
    rows = []
    for partition, chrom in [("train", "chr3"), ("validation", "chr2")]:
        for i in range(20):
            label = i % 2
            row = {key: 0.0 for key in GEOMETRY + DEGREE}
            row.update(signed_offset=float(label), label=label, partition=partition, chrom=chrom,
                       context="strict", slice=chrom, candidate_index=i)
            rows.append(row)
    frame = pd.DataFrame(rows)
    scores, predictions = fit_validation_baselines(frame)
    assert scores.set_index("baseline").loc["geometry", "auroc"] == 1
    changed = frame.copy()
    changed.loc[changed.partition == "validation", "label"] ^= 1
    _, other = fit_validation_baselines(changed)
    np.testing.assert_allclose(predictions.probability, other.probability)
    with pytest.raises(ValueError, match="overlap"):
        fit_validation_baselines(frame.assign(chrom="chr1"))
    with pytest.raises(ValueError, match="held-out"):
        fit_validation_baselines(frame.replace({"partition": {"validation": "test"}}))


@pytest.mark.parametrize("objective", ["edge_masking", "junction_repair"])
def test_optional_loader_receipt_does_not_change_native_arrays(tmp_path, objective):
    from graph.features import build_oid_metadata_from_segments
    from graph.slicing import build_global_index
    from training.pretrain import load_slice

    n = 152
    segments = pd.DataFrame(dict(id=np.arange(n), name=[str(i) for i in range(n)],
                                LN=100, SN="GRCh38#0#chr1", SO=np.arange(n) * 100, SR=0))
    left = list(range(n - 1)) + list(range(0, n - 2, 3))
    right = list(range(1, n)) + list(range(2, n, 3))
    links = pd.DataFrame(dict(from_seg=[str(i) for i in left], to_seg=[str(i) for i in right],
                             from_orient="+", to_orient="+"))
    edges = pd.DataFrame(dict(u_oid=np.arange(0, 80, 2), v_oid=np.arange(2, 82, 2), label=1))
    paths = {}
    for kind, data in [("segments", segments), ("links", links), ("edge_pred", edges)]:
        path = tmp_path / (kind + ".csv")
        data.to_csv(path, index=False)
        paths[kind + "_path"] = str(path)
    row = pd.Series(dict(name="window", target_sn="GRCh38#0#chr1", closure="strict", **paths))
    index, _ = build_global_index(segments)
    md = build_oid_metadata_from_segments(segments, index)
    args = Namespace(objective=objective, canonical_conflict_policy="error", seed=42,
                     split_seed=20260806, pop_cond=False, use_edge_features=False)
    original = load_slice(row, index, md, segments, args)
    receipt = {}
    audited = load_slice(row, index, md, segments, args, audit_out=receipt)
    assert original is not None and receipt["exclusion"] == "retained"
    for key in original:
        if isinstance(original[key], np.ndarray):
            np.testing.assert_array_equal(original[key], audited[key])
    assert original.keys() == audited.keys()
    assert receipt["n_candidate_node_filter_exclusions"] == 0

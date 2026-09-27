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
@pytest.mark.parametrize("mixed_storage", [False, True])
def test_optional_loader_receipt_does_not_change_native_arrays(tmp_path, objective, mixed_storage):
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
    if mixed_storage:
        # Make one canonical forward handle unavailable, while preserving
        # enough unaffected groups to exercise a retained native slice.
        reverse = (np.asarray(left) == 43) | (np.asarray(right) == 43)
        links.loc[reverse, ["from_seg", "to_seg"]] = links.loc[reverse, ["to_seg", "from_seg"]].to_numpy()
        links.loc[reverse, ["from_orient", "to_orient"]] = "-"
    edges = pd.DataFrame(dict(u_oid=np.arange(0, 120, 2), v_oid=np.arange(2, 122, 2), label=1))
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
    if mixed_storage:
        assert receipt["n_candidate_node_filter_exclusions"] > 0
        if objective == "junction_repair":
            validate_groups(audited)
            assert receipt["n_valid_partner_rows_excluded"] > 0
        else:
            assert receipt["n_candidate_node_filter_exclusions"] == 2
    else:
        assert receipt["n_candidate_node_filter_exclusions"] == 0


def test_raw_node_controls_use_visible_features_and_no_validation_label_fit():
    from tasks.transfer.junction_readiness import GEOMETRY, DEGREE
    sd = native_fixture()
    args = Namespace(seed=42, batch_size=4, orientation_rope=False, pop_cond=False,
                     drop_edge=False, drop_edge_rate=.1, include_node_controls=True)
    rows = feature_rows(sd, args, 'train')
    np.testing.assert_array_equal(rows.node_4_u, rows.degree_u)
    assert rows.node_6_u.eq(0).all()
    records = []
    for partition, chrom in [('train', 'chr3'), ('validation', 'chr2')]:
        for i in range(80):
            row = {key: 0. for key in GEOMETRY + DEGREE}
            row.update(node_0_u=float(i % 2), node_0_v=float(i % 2), label=i % 2,
                       partition=partition, chrom=chrom, context='1hop', slice=chrom,
                       candidate_index=i)
            records.append(row)
    data = pd.DataFrame(records)
    scores, predictions = fit_validation_baselines(data)
    assert set(scores.baseline) >= {'node_inputs_linear', 'node_inputs_boosting'}
    data.loc[data.partition.eq('validation'), 'label'] ^= 1
    _, changed = fit_validation_baselines(data)
    np.testing.assert_array_equal(predictions.probability, changed.probability)


def test_raw_nt_control_preserves_all_endpoint_interactions_without_fragmentation():
    import warnings
    sd = native_fixture()
    extra = np.arange(16 * 512, dtype=np.float32).reshape(16, 512) / 1000
    sd['node_feats'] = np.column_stack([sd['node_feats'], extra])
    args = Namespace(seed=42, batch_size=4, orientation_rope=False, pop_cond=False,
                     drop_edge=False, drop_edge_rate=.1, include_node_controls=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error', pd.errors.PerformanceWarning)
        rows = feature_rows(sd, args, 'train')
    idx = sd['train_idx']
    u, v = sd['query_u'][idx], sd['query_v'][idx]
    assert len([c for c in rows if c.startswith('node_')]) == 4 * 519
    np.testing.assert_allclose(rows.node_518_u, extra[u, -1])
    np.testing.assert_allclose(rows.node_518_v, extra[v, -1])
    np.testing.assert_allclose(rows.node_518_product, extra[u, -1] * extra[v, -1])
    np.testing.assert_allclose(rows.node_518_absdiff, abs(extra[u, -1] - extra[v, -1]))

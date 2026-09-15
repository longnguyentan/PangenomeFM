import numpy as np
import pandas as pd
import pytest
from tasks.entex.prepare import aggregate_ccre, validate_as
from tasks.entex.mapping import map_loci, aggregate_features


def measurements():
    return pd.DataFrame(
        dict(
            chr=["chr1"] * 3,
            start=[10, 10, 30],
            end=[20, 20, 40],
            region_id=["a_dELS", "a_dELS", "b_dELS"],
            experiment_accession=["e1", "e2", "e1"],
            donor=["d1", "d2", "d1"],
            tissue=["t1"] * 3,
            assay=["a"] * 3,
            hap1_count=[10] * 3,
            hap2_count=[10] * 3,
            p_betabinom=[0.01, 0.5, 0.8],
            imbalance_significance=[1, 0, 0],
        )
    )


def test_union_and_measured_negatives():
    x = measurements()
    out = aggregate_ccre(pd.concat([x, x.iloc[:1]]))
    assert out.label.tolist() == [1, 0]
    assert out.n_informative_experiments.tolist() == [2, 1]
    assert out.donor_count.tolist() == [2, 1]
    assert out.n_as_experiments.tolist() == [1, 0]


def test_conflicting_duplicate_rejected():
    x = measurements()
    duplicate = x.iloc[:1].copy()
    duplicate["imbalance_significance"] = 0
    with pytest.raises(ValueError, match="Conflicting"):
        aggregate_ccre(pd.concat([x, duplicate]))


@pytest.mark.parametrize(
    "column,value",
    [
        ("start", -1),
        ("end", 10),
        ("chr", "1"),
        ("imbalance_significance", 2),
        ("p_betabinom", float("nan")),
        ("hap1_count", -1),
    ],
)
def test_bad_input(column, value):
    x = measurements()
    x.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_as(x)


def test_overlap_boundaries_and_pooling():
    loci = pd.DataFrame(
        dict(
            chrom=["chr1"] * 3,
            start=[5, 10, 30],
            end=[20, 11, 31],
            locus_id=["a", "b", "c"],
        )
    )
    nodes = pd.DataFrame(
        dict(chrom=["chr1"] * 2, SO=[0, 10], LN=[10, 20], segid=[3, 4])
    )
    edges = map_loci(loci, nodes)
    assert edges.to_records(index=False).tolist() == [
        ("a", 3, 5),
        ("a", 4, 10),
        ("b", 4, 1),
    ]
    x = aggregate_features(
        loci.iloc[:2],
        edges,
        np.array([4, 3]),
        np.array([[6.0], [0.0]]),
        "length_weighted",
    )
    np.testing.assert_allclose(x[:, 0], [4, 6])
    x = aggregate_features(
        loci.iloc[:2], edges, np.array([3, 4]), np.array([[0.0], [6.0]]), "mean"
    )
    np.testing.assert_allclose(x[:, 0], [3, 6])
    with pytest.raises(ValueError, match="partial-locus"):
        aggregate_features(
            loci.iloc[:2], edges, np.array([3]), np.array([[0.0]]), "mean"
        )


def test_split_and_complete_universe():
    from tasks.entex.probe import check_splits, complete_loci

    loci = pd.DataFrame(dict(locus_id=["a", "b"], chrom=["chr1", "chr2"]))
    check_splits(
        loci,
        [
            dict(test=["chr1"], validation=["chr2"]),
            dict(test=["chr2"], validation=["chr1"]),
        ],
    )
    with pytest.raises(ValueError, match="overlap"):
        check_splits(loci, [dict(test=["chr1", "chr2"], validation=["chr1"])])
    edges = pd.DataFrame(dict(locus_id=["a", "a", "b"], segid=[1, 2, 1]))
    assert complete_loci(loci, edges, {1}).locus_id.tolist() == ["b"]


def test_probe_and_paired_smoke(tmp_path):
    from scripts.server.run_ccre_frozen_probe_fold import evaluate_feature_sets
    from tasks.entex.probe import FEATURES
    from evaluation.modality_factorial import build_modality_factorial
    from tasks.entex.analyze import paired, BASE, FULL

    rng = np.random.default_rng(42)
    y = np.tile([0, 1], 60)
    components = {
        k: rng.normal(size=(120, 3)).astype("float32")
        for k in [
            "coordinate",
            "sequence_kmer",
            "frozen_sequence_fm",
            "frozen_pangenomefm",
        ]
    }
    features = build_modality_factorial(components, include_external_sequence=True)
    m, _, p = evaluate_feature_sets(
        segids=np.arange(120),
        chromosomes=np.repeat(["chr1", "chr2", "chr3"], 40),
        labels=y,
        features={k: features[k] for k in FEATURES},
        test_chrs={"chr1"},
        val_chrs={"chr2"},
        seed=42,
    )
    assert len(m) == 7 and len(p) == 280
    assert m.n_test.eq(40).all() and set(p.chromosome) == {"chr1"}
    m["fold"] = "fold_a"
    m["seed"] = 42
    result = paired(m, BASE, 100, 42)
    assert np.isnan(result["ci95_low"])
    assert result["mean"] == pytest.approx(
        float(
            m.loc[m.feature_set.eq(FULL), "auprc"].iloc[0]
            - m.loc[m.feature_set.eq(BASE), "auprc"].iloc[0]
        )
    )
    m.to_csv(tmp_path / "synthetic_smoke_metrics.csv", index=False)


def test_stream_audit(tmp_path):
    from tasks.entex.prepare import inspect_source

    source = tmp_path / "input.tsv"
    measurements().to_csv(source, sep="\t", index=False)
    a = inspect_source(source, tmp_path / "cache", 1)
    assert a["raw_row_count"] == 3 and a["unique_genomic_loci"] == 2
    assert all(x == 0 for x in a["missing_values"].values())
    assert len(pd.read_parquet(tmp_path / "cache/input.tsv.parquet")) == 3


def test_analysis_outputs(tmp_path, monkeypatch):
    import sys
    from tasks.entex.analyze import main, BASE, FULL, CT

    root = tmp_path / "synthetic_only"
    run = root / "fold_a/seed_42/strict"
    run.mkdir(parents=True)
    rows = []
    pred = []
    for key in [BASE, FULL, CT]:
        rows.append(
            dict(
                fold="fold_a",
                seed=42,
                closure="strict",
                feature_set=key,
                auprc=0.7,
                auroc=0.7,
                balanced_accuracy=0.5,
                f1=0.5,
                precision=0.5,
                recall=0.5,
            )
        )
        for i in range(6):
            pred.append(
                dict(
                    fold="fold_a",
                    seed=42,
                    closure="strict",
                    feature_set=key,
                    locus_id=str(i),
                    chrom="chr1",
                    start=i * 10,
                    end=i * 10 + 1,
                    y_true=i % 2,
                    p_calibrated=0.2 + 0.1 * i,
                )
            )
    pd.DataFrame(rows).to_csv(run / "metrics.csv", index=False)
    pd.DataFrame(pred).to_parquet(run / "predictions.parquet", index=False)
    c = tmp_path / "complexity.tsv"
    pd.DataFrame(
        dict(
            chromosome=["chr1"] * 3,
            start=[0, 20, 40],
            end=[20, 40, 60],
            context=["strict"] * 3,
            locus_complexity_category=["low", "medium", "high"],
        )
    ).to_csv(c, sep="\t", index=False)
    out = tmp_path / "analysis"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze",
            "--probe-root",
            str(root),
            "--out-dir",
            str(out),
            "--complexity",
            str(c),
            "--n-bootstrap",
            "100",
        ],
    )
    main()
    assert len(pd.read_csv(out / "complexity_per_run.csv")) == 6
    assert len(list(out.glob("*.svg"))) == 3
    assert pd.read_csv(out / "paired_gains.csv").ci95_low.isna().all()


def test_single_class_chromosome_allowed():
    x = measurements().iloc[:2]
    assert len(aggregate_ccre(x, require_both_classes=False)) == 1
    with pytest.raises(ValueError, match="both classes"):
        aggregate_ccre(x)


def test_exact_exposure_matching_is_stable_and_partition_local():
    from tasks.entex.sensitivity import match_exposure

    loci = pd.DataFrame(
        dict(
            locus_id=list("abcdefghijk"),
            chrom=["chr1"] * 7 + ["chr2"] * 4,
            n_informative_experiments=[1, 1, 1, 1, 2, 2, 3, 1, 1, 2, 2],
            label=[1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
        )
    )
    matched, balance = match_exposure(loci, 42)
    shuffled, _ = match_exposure(loci.sample(frac=1, random_state=8), 42)
    assert matched.locus_id.tolist() == shuffled.locus_id.tolist()
    assert matched.label.mean() == 0.5
    assert len(matched) == 6
    counts = (
        matched.groupby(["chrom", "n_informative_experiments", "label"])
        .size()
        .unstack()
    )
    assert counts[0].equals(counts[1])
    assert balance.selected_per_class.sum() * 2 == len(matched)
    local, _ = match_exposure(loci.loc[loci.chrom.eq("chr1")], 42)
    assert (
        local.locus_id.tolist()
        == matched.loc[matched.chrom.eq("chr1"), "locus_id"].tolist()
    )
    assert "g" not in set(matched.locus_id)  # no measurable control at exposure=3
    with pytest.raises(ValueError, match="overlapping exposure"):
        match_exposure(loci.loc[loci.label.eq(1)], 42)


def test_assay_labels_are_recomputed(tmp_path):
    from tasks.entex.sensitivity import prepare

    x = measurements()
    x["assay"] = ["HM-ChIP-seq_H3K27ac", "TF-ChIP-seq_CTCF", "HM-ChIP-seq_H3K27ac"]
    extra = x.iloc[[2]].copy()
    extra["assay"] = "TF-ChIP-seq_CTCF"
    extra["experiment_accession"] = "ctcf2"
    extra["imbalance_significance"] = 1
    x = pd.concat([x, extra], ignore_index=True)
    cache = tmp_path / "measurements.parquet"
    x.to_parquet(cache, index=False)
    # Primary preview requires overlapping exposure support.
    primary = tmp_path / "primary.parquet"
    pd.DataFrame(
        dict(
            locus_id=["x", "y"],
            chrom=["chr1"] * 2,
            n_informative_experiments=[2, 2],
            label=[0, 1],
        )
    ).to_parquet(primary, index=False)
    out = tmp_path / "out"
    settings = {
        "sensitivities": {
            "assays": {"h3k27ac": "HM-ChIP-seq_H3K27ac", "ctcf": "TF-ChIP-seq_CTCF"},
            "matching_seed": 42,
            "matching_keys": ["chrom", "n_informative_experiments"],
        }
    }
    prepare(cache, primary, out, settings)
    h = pd.read_parquet(out / "h3k27ac_loci.parquet")
    c = pd.read_parquet(out / "ctcf_loci.parquet")
    assert h.label.tolist() == [1, 0]
    assert c.label.tolist() == [0, 1]
    assert h.n_informative_experiments.eq(1).all()
    assert c.n_informative_experiments.eq(1).all()
    with pytest.raises(FileExistsError):
        prepare(cache, primary, out, settings)


def test_normalized_ap_and_matched_comparators():
    from tasks.entex.analyze import (
        ranking_metrics,
        validate_comparator_loci,
        BASE,
        FULL,
        paired,
    )

    y = pd.Series([0, 0, 0, 1])
    assert ranking_metrics(y, pd.Series([0.1, 0.1, 0.1, 0.1]))["normalized_ap"] == 0
    perfect = ranking_metrics(y, pd.Series([0.1, 0.2, 0.3, 0.9]))
    assert perfect["normalized_ap"] == 1 and perfect["auroc"] == 1
    assert np.isnan(
        ranking_metrics(pd.Series([1, 1]), pd.Series([0.2, 0.8]))["normalized_ap"]
    )
    p = pd.DataFrame(
        dict(
            locus_id=["a", "b", "a", "b"],
            feature_set=[BASE, BASE, FULL, FULL],
            y_true=[0, 1, 0, 1],
        )
    )
    validate_comparator_loci(p)
    with pytest.raises(ValueError, match="same loci"):
        validate_comparator_loci(p.iloc[:3])
    p.loc[2, "y_true"] = 1
    with pytest.raises(ValueError, match="labels differ"):
        validate_comparator_loci(p)
    scores = pd.DataFrame(
        dict(
            fold=["a", "a", "b", "b"],
            seed=[42] * 4,
            feature_set=[BASE, FULL] * 2,
            auroc=[0.5, 0.6, 0.7, 0.8],
        )
    )
    assert paired(scores, BASE, 100, 42, metric="auroc")["mean"] == pytest.approx(0.1)

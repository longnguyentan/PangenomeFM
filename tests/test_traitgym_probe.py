import json

import numpy as np
import pandas as pd
import pytest

from evaluation.modality_factorial import build_modality_factorial
from scripts.server.audit_traitgym_coverage import normalize
from scripts.server.run_ccre_frozen_probe_fold import evaluate_feature_sets
from tasks.entex.prepare import fingerprint
from tasks.transfer.traitgym import (align_component, author_variant_scores, validate_examples,
    validate_nt_provenance, weighted_chromosome_ap)
from tasks.transfer.traitgym_report import METRICS, expected_test_support, markdown_contrasts, replay_run, summarize


def fixture():
    source = pd.DataFrame([dict(chrom=f"chr{c}", pos=100 + i, ref="A", alt="C", label=int(i == 0),
                                match_group=f"group{c}") for c in range(1, 6) for i in range(10)])
    examples = normalize(source)
    overlaps = examples[["locus_id"]].assign(segid=np.arange(len(examples)), overlap_bp=1)
    folds = [dict(name=f"fold_{c}", test=[f"chr{c}"], validation=[f"chr{c % 5 + 1}"])
             for c in range(1, 6)]
    return source, examples, overlaps, folds


def test_original_variants_groups_and_split_guards():
    source, examples, overlaps, folds = fixture()
    assert validate_examples(examples, source, overlaps, folds)["n"] == 50
    with pytest.raises(AssertionError):
        validate_examples(examples.iloc[1:], source, overlaps, folds)
    bad = source.copy()
    bad.loc[0, "chrom"] = "chr2"
    bad.loc[0, "pos"] = 300
    with pytest.raises(ValueError, match="single-chromosome"):
        validate_examples(normalize(bad), bad, overlaps, folds)
    with pytest.raises(ValueError, match="exactly one"):
        validate_examples(examples, source, overlaps.iloc[1:], folds)
    with pytest.raises(ValueError, match="overlap"):
        validate_examples(examples, source, overlaps, [dict(test=["chr1"], validation=["chr1"]), *folds[1:]])


def test_alleles_share_features_without_collapsing_variants():
    _, examples, overlaps, _ = fixture()
    extra = examples.iloc[[0]].copy()
    extra["variant_id"] = extra.variant_id.str.replace(":A:C", ":A:G")
    extra["alt"] = "G"
    repeated = pd.concat([examples, extra], ignore_index=True)
    values = np.arange(100).reshape(50, 2)
    aligned = align_component(repeated, overlaps, np.arange(50), values)
    assert aligned.shape == (51, 2)
    np.testing.assert_array_equal(aligned[0], aligned[-1])
    # Reversing cache row order must not change feature alignment.
    np.testing.assert_array_equal(aligned, align_component(repeated, overlaps, np.arange(50)[::-1], values[::-1]))
    with pytest.raises(ValueError, match="Missing segment"):
        align_component(repeated, overlaps, np.arange(49), values[:-1])


def test_chromosome_weighting_has_explicit_denominator():
    frame = pd.DataFrame(dict(chromosome=["chr1", "chr2"], n=[10, 30], auprc=[.2, .6]))
    assert weighted_chromosome_ap(frame) == pytest.approx(.5)
    with pytest.raises(ValueError, match="defined"):
        weighted_chromosome_ap(frame.assign(auprc=[np.nan, .6]))
    assert expected_test_support({"chromosome_counts": {"chr1": 10, "chr6": 20}},
                                 ["chr1", "chr6", "chr21"]) == (["chr1", "chr6"], 30)


def test_merged_nt_cache_shard_provenance(tmp_path):
    cfg = json.loads(open("configs/entex_v1.json").read())
    shard = dict(model_name=cfg["nt_model"], resolved_revision=cfg["nt_revision"],
        full_segments_sha256=cfg["full_segments_sha256"], maximum_token_length=cfg["nt_max_length"],
        maximum_raw_bases=cfg["nt_max_bases"], model_parameters_frozen=True, fine_tuned=False,
        pooling="mean final hidden state over non-special, non-padding tokens")
    sidecar = tmp_path / "shard_0.npz.audit.json"
    sidecar.write_text(json.dumps(shard))
    audit = dict(output_sha256="test", source_shards=[dict(path="/original/shard_0.npz",
                  audit_sha256=fingerprint(sidecar)["sha256"])])
    assert len(validate_nt_provenance(tmp_path / "merged.npz", audit, cfg, "test")) == 1
    with pytest.raises(ValueError, match="checksum"):
        validate_nt_provenance(tmp_path / "merged.npz", audit, cfg, "modified")
    sidecar.write_text(json.dumps(dict(shard, fine_tuned=True)))
    with pytest.raises(ValueError, match="Stale"):
        validate_nt_provenance(tmp_path / "merged.npz", audit, cfg, "test")
    audit["source_shards"][0]["audit_sha256"] = fingerprint(sidecar)["sha256"]
    with pytest.raises(ValueError, match="provenance"):
        validate_nt_provenance(tmp_path / "merged.npz", audit, cfg, "test")


def test_author_scores_require_pinned_hash_and_join_original_variant_identity(tmp_path):
    original, examples, _, _ = fixture()
    path = tmp_path / "scores.parquet"
    score = np.arange(len(original), dtype=float) - 20
    pd.DataFrame({"score": score}).to_parquet(path, index=False)
    digest = fingerprint(path)["sha256"]
    matrix, _ = author_variant_scores(path, digest, original, examples.iloc[::-1])
    np.testing.assert_array_equal(matrix[:, 0], score[::-1])
    np.testing.assert_array_equal(matrix[:, 1], abs(score[::-1]))
    with pytest.raises(ValueError, match="Stale"):
        author_variant_scores(path, "not-the-pinned-source", original, examples)
    with pytest.raises(ValueError, match="row-order"):
        author_variant_scores(path, digest, original.iloc[::-1], examples)


def test_native_probe_replay_and_tamper_detection(tmp_path):
    _, examples, _, _ = fixture()
    plan = json.loads(open("configs/traitgym_locus_prior_20260927.json").read())
    rng = np.random.default_rng(19)
    components = {k: rng.normal(size=(50, 3)) for k in
                  ["coordinate", "sequence_kmer", "frozen_sequence_fm", "frozen_pangenomefm", "topology_control"]}
    features = build_modality_factorial(components, include_external_sequence=True, include_topology_control=True)
    metrics, per_chr, predictions = evaluate_feature_sets(segids=np.arange(50), chromosomes=examples.chrom,
        labels=examples.label.to_numpy(), features={k: features[k] for k in plan["feature_sets"]},
        test_chrs={"chr1"}, val_chrs={"chr2"}, seed=42, probe_max_iter=4000)
    from sklearn.metrics import balanced_accuracy_score
    metrics["balanced_accuracy"] = [balanced_accuracy_score(p.y_true, p.y_pred) for f in metrics.feature_set
        for p in [predictions.loc[predictions.feature_set.eq(f)]]]
    metrics["chromosome_weighted_auprc"] = [weighted_chromosome_ap(per_chr.loc[per_chr.feature_set.eq(f)])
                                           for f in metrics.feature_set]
    metrics["normalized_ap"] = (metrics.auprc - metrics.positive_fraction) / (1 - metrics.positive_fraction)
    predictions = predictions.merge(examples.assign(segid=np.arange(50)), on="segid", validate="many_to_one")
    for f in [metrics, predictions]:
        for k, v in dict(dataset="complex_traits", context="strict", fold="fold_a", seed=42,
                         task="traitgym_locus_prior").items():
            f[k] = v
    metrics.to_csv(tmp_path / "metrics.csv", index=False)
    predictions.to_parquet(tmp_path / "predictions.parquet", index=False)
    (tmp_path / "audit.json").write_text(json.dumps(dict(status="complete", n_excluded=0,
        predictions=fingerprint(tmp_path / "predictions.parquet"))))
    checked, identity = replay_run(tmp_path, plan, ["chr1"])
    assert len(checked) == 9 and len(identity) == 10
    absolute, contrasts, paired = summarize(checked, plan)
    assert len(absolute) == len(METRICS) * 9 and len(paired) == 12
    assert contrasts.ci95_low.isna().all()  # A smoke fold does not provide a fold CI.
    assert "T_given_CSH" in markdown_contrasts(contrasts)
    assert checked.probe_converged.all()
    metrics.loc[0, "auprc"] += .1
    metrics.to_csv(tmp_path / "metrics.csv", index=False)
    with pytest.raises(ValueError, match="differs from replay"):
        replay_run(tmp_path, plan, ["chr1"])

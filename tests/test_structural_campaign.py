import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.paired_inference import bh_adjust, fold_sign_flip
from tasks.transfer.scaling import nested_manifests, pending_tasks
from tasks.transfer.sv_strata import annotate, run_metrics
from tasks.transfer.donors import overlap_table, parse_gt
from tasks.transfer.report import paired_rows
from tasks.transfer.scaling_evaluate import commands
from scripts.server.run_ccre_frozen_probe_matrix import ProbeJob
from argparse import Namespace
from tasks.entex.analyze import BASE, FULL


def test_nested_scaling_preserves_all_heldout_rows_and_context_pairs():
    rows = []
    for chrom in ["chr1", "chr2", "chr3", "chr4"]:
        for position in range(16):
            for context in ["strict", "1hop"]:
                rows.append(
                    dict(
                        name=f"{chrom}_{position}_{context}",
                        target_sn=f"GRCh38#0#{chrom}",
                        start=position * 10,
                        end=(position + 1) * 10,
                        closure=context,
                    )
                )
    original = pd.DataFrame(rows)
    fold = dict(test=["chr1"], validation=["chr2"])
    first = nested_manifests(original, fold)
    second = nested_manifests(original.sample(frac=1, random_state=1), fold)
    previous = set()
    for fraction, frame in first.items():
        assert set(frame.name) == set(second[fraction].name)
        assert previous.issubset(set(frame.name))
        assert frame.groupby(["target_sn", "start"]).closure.nunique().eq(2).all()
        assert len(frame.loc[frame.target_sn.str.endswith(("chr1", "chr2"))]) == 64
        assert len(frame.loc[frame.target_sn.str.endswith("chr3")]) == 32 * fraction
        previous = set(frame.name)
    pd.testing.assert_frame_equal(first[1.0], original)


def test_scaling_rejects_leaking_fold():
    frame = pd.DataFrame(
        dict(name=["a"], target_sn=["chr1"], start=[0], end=[10], closure=["strict"])
    )
    with pytest.raises(ValueError, match="Overlapping"):
        nested_manifests(frame, dict(test=["chr1"], validation=["chr1"]))


def test_exact_fold_signflip_and_bh():
    frame = pd.DataFrame(dict(fold=np.repeat(list("abcde"), 3), gain=[0.1] * 15))
    assert fold_sign_flip(frame) == 0.0625
    # Repeating seeds does not pretend to increase the number of independent folds.
    assert fold_sign_flip(pd.concat([frame] * 2)) == 0.0625
    np.testing.assert_allclose(
        bh_adjust([0.01, 0.04, 0.03, np.nan])[:3], [0.03, 0.04, 0.04]
    )
    assert np.isnan(bh_adjust([0.01, 0.04, 0.03, np.nan])[-1])


def test_sv_class_is_target_not_a_discrimination_stratum(tmp_path):
    path = tmp_path / "complexity.tsv"
    pd.DataFrame(
        dict(
            context=["strict"] * 2,
            chromosome=["chr1"] * 2,
            start=[0, 10],
            end=[10, 20],
            locus_complexity_category=["low", "high"],
        )
    ).to_csv(path, sep="\t", index=False)
    raw = pd.DataFrame(
        dict(
            example_id=[0, 1],
            chrom=["chr1"] * 2,
            start0=[1, 2],
            end0=[11, 3],
            svtype=["DEL", "INS"],
            binary_svtype_label=[0, 1],
            alt_allele_count=[1, 2],
            length_bin=["bp[50,100)"] * 2,
            af_bin=["af[0,.01)"] * 2,
        )
    )
    meta = annotate(raw, path)
    assert meta.complexity.tolist() == ["high", "low"]
    pred = pd.DataFrame(
        dict(
            example_id=[0, 1],
            fold=["fold_a"] * 2,
            seed=[42] * 2,
            closure=["strict"] * 2,
            feature_set=["C+S"] * 2,
            y_true=[0, 1],
            p_calibrated=[0.1, 0.9],
            threshold=[0.5] * 2,
            y_pred=[0, 1],
        )
    )
    cfg = json.loads(Path("configs/structural_mechanism_v1.json").read_text())
    metrics = run_metrics(pred, meta, cfg)
    by_type = metrics.loc[metrics.family.eq("sv_type")]
    assert by_type.auprc.isna().all() and by_type.auroc.isna().all()
    assert by_type.class_recall.eq(1).all()
    assert not by_type.sufficiently_powered.any()


def test_donor_alias_prevents_false_external_claim():
    membership = pd.DataFrame(
        dict(sample=["HG002", "HG00733", "other"], hprc_r2=[True, True, False])
    )
    table = overlap_table(
        ["NA24385", "HG00733", "other"], membership, {"NA24385": "HG002"}
    )
    assert table.in_hprc_pretraining_cohort.tolist() == [True, True, False]
    with pytest.raises(ValueError, match="duplicate biological"):
        overlap_table(["HG002", "NA24385"], membership, {"NA24385": "HG002"})


def test_missing_and_unphased_genotypes_are_not_haplotype_reference_calls():
    assert parse_gt("1|.") == (1, -1)
    assert parse_gt(".|0") == (-1, 0)
    assert parse_gt("1/0") == (-1, -1)
    assert parse_gt("1") == (1, -1)
    with pytest.raises(ValueError, match="biallelic"):
        parse_gt("2|0")


def test_report_pairs_counts_and_undefined_classes():
    frame = pd.DataFrame(
        dict(
            fold=["a", "a", "b", "b"],
            seed=[42] * 4,
            feature_set=[BASE, FULL] * 2,
            auprc=[0.5, 0.6, np.nan, np.nan],
            n=[100] * 4,
            positive_prevalence=[0.5, 0.5, 1, 1],
        )
    )
    result = paired_rows(frame, "auprc")
    assert result.gain.iloc[0] == pytest.approx(0.1)
    assert np.isnan(result.gain.iloc[1])
    with pytest.raises(ValueError, match="Missing paired"):
        paired_rows(frame.iloc[:-1], "auprc")
    frame.loc[0, "n"] = 99
    with pytest.raises(ValueError, match="counts/prevalence"):
        paired_rows(frame, "auprc")


def test_reuse_scaling_requires_identical_data_command_and_checkpoint(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("same manifest")
    b.write_text(a.read_text())
    out = tmp_path / "checkpoint"
    (out / "run_001").mkdir(parents=True)
    (out / "run_001/ckpt_strict.pt").write_bytes(b"fixture")
    status_dir = tmp_path / "execution/task_status"
    status_dir.mkdir(parents=True)
    cmd = [
        "python",
        "-m",
        "training.pretrain",
        "--manifest",
        str(a),
        "--out_dir",
        str(out),
    ]
    (status_dir / "pilot.json").write_text(
        json.dumps(dict(task_id="pilot", status="complete", return_code=0, command=cmd))
    )
    current = cmd.copy()
    current[4] = str(b)
    tasks = [dict(id="pilot", command=current)]
    todo, done = pending_tasks(tasks, tmp_path / "execution")
    assert not todo and len(done) == 1
    b.write_text("changed manifest")
    with pytest.raises(ValueError, match="manifest changed"):
        pending_tasks(tasks, tmp_path / "execution")


def test_scaling_evaluation_reuses_full_graph_and_original_probe_interfaces():
    args = Namespace(
        resource_root=Path("resource"),
        out_root=Path("new_results"),
        entex_root=Path("entex"),
        checkpoint_root=Path("scaled"),
        device="cpu",
    )
    job = ProbeJob("fold_a", ("chr1",), ("chr2",), 42, "strict")
    plan = commands(args, job, Path("scaled/checkpoint.pt"))
    for command in plan.values():
        assert (
            command[command.index("--manifest") + 1]
            == "resource/data/benchmarks/hprc_r2_pretrain_5mb_paired/manifest.csv"
        )
        assert not any("smoke" in x for x in command)
    assert plan["sv"][plan["sv"].index("--checkpoint") + 1] == "scaled/checkpoint.pt"
    assert "--measurements" in plan["ctcf"] and "--tasks" not in plan["ctcf"]
    assert plan["ctcf"][plan["ctcf"].index("--results-root") + 1] == "scaled"

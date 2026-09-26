"""Audit all scaling checkpoints and summarize fixed held-out reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluation.external import _build_model_from_checkpoint, _namespace_from_checkpoint
from scripts.server.aggregate_server_results import metric_row
from scripts.server.run_ccre_frozen_probe_fold import validate_checkpoint_holdout
from scripts.server.run_ccre_frozen_probe_matrix import build_jobs, checkpoint_for
from tasks.entex.analyze import estimate
from tasks.entex.prepare import fingerprint
from tasks.transfer.scaling import FRACTIONS


def target_digest(frame: pd.DataFrame) -> str:
    """Include labels, partitions and oriented endpoints, independent of row order."""
    columns = [
        "dataset",
        "slice",
        "target_sn",
        "closure",
        "split",
        "u_local",
        "v_local",
        "y_true",
    ]
    if frame.empty or frame[columns].isna().any().any():
        raise ValueError("Empty or missing held-out targets")
    if frame.duplicated(columns[:-1]).any():
        raise ValueError("Duplicate held-out targets")
    if not set(frame.y_true).issubset({0, 1}):
        raise ValueError("Invalid reconstruction labels")
    content = frame[columns].sort_values(columns).to_csv(index=False)
    return hashlib.sha256(content.encode()).hexdigest()


def validate_scales(frame: pd.DataFrame) -> None:
    keys = ["fold", "seed", "context"]
    if frame.duplicated(keys + ["fraction"]).any():
        raise ValueError("Duplicate scaling run")
    if not all(set(g.fraction) == set(FRACTIONS) for _, g in frame.groupby(keys)):
        raise ValueError("Incomplete scaling fractions")
    for _, group in frame.groupby(keys):
        for column in [
            "test_targets_sha256",
            "validation_targets_sha256",
            "arguments_sha256",
        ]:
            if group[column].nunique() != 1:
                raise ValueError(f"Scaling comparison changed {column}")
    if frame.n_parameters.nunique() != 1 or frame.architecture_sha256.nunique() != 1:
        raise ValueError("Scaling architecture changed")


def summaries(frame: pd.DataFrame, n_bootstrap: int, seed: int):
    absolute, paired = [], []
    metrics = ["auprc", "auroc", "mean_window_auprc"]
    for context, context_frame in frame.groupby("context"):
        full = context_frame.loc[context_frame.fraction.eq(1)].set_index(
            ["fold", "seed"]
        )
        for fraction, group in context_frame.groupby("fraction"):
            for metric in metrics:
                meta = dict(context=context, fraction=fraction, metric=metric)
                absolute.append(
                    dict(**meta, **estimate(group, metric, n_bootstrap, seed))
                )
                delta = group.set_index(["fold", "seed"])[metric] - full[metric]
                if delta.isna().any():
                    raise ValueError("Incomplete paired scaling comparison")
                paired.append(
                    dict(
                        **meta,
                        comparator_fraction=1.0,
                        **estimate(
                            delta.rename("gain").reset_index(),
                            "gain",
                            n_bootstrap,
                            seed,
                        ),
                    )
                )
    return pd.DataFrame(absolute), pd.DataFrame(paired)


def figures(absolute: pd.DataFrame, paired: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained")
    for context, color in [("strict", "#245a81"), ("1hop", "#be6831")]:
        for ax, data in zip(axes, [absolute, paired]):
            selected = data.loc[
                data.context.eq(context) & data.metric.eq("auprc")
            ].sort_values("fraction")
            ax.plot(
                selected.fraction, selected["mean"], "o-", color=color, label=context
            )
            ax.vlines(
                selected.fraction, selected.ci95_low, selected.ci95_high, color=color
            )
            ax.set_xscale("log", base=2)
            ax.set(
                xticks=FRACTIONS,
                xticklabels=["12.5%", "25%", "50%", "100%"],
                xlabel="Requested fraction of training windows",
            )
    axes[0].set(
        ylabel="Held-out pooled AUPRC", title="Reconstruction at fixed test targets"
    )
    axes[0].legend(frameon=False)
    axes[1].axhline(0, color=".5", lw=0.7)
    axes[1].set(
        ylabel="Paired AUPRC difference from 100%",
        title="Newly trained 100% comparator",
    )
    fig.suptitle("Five chromosome folds × three seeds; 95% fold/seed bootstrap CI")
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(out / f"intrinsic_scaling.{suffix}", dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("configs/server_full_multicohort_20260806.json"),
    )
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260924)
    args = ap.parse_args()
    jobs = build_jobs(json.loads(args.config.read_text()))
    if len(jobs) != 30:
        raise ValueError("Expected all five folds, three seeds and two contexts")
    rows, sources = [], []
    canonical = json.loads(Path("configs/entex_v1.json").read_text())
    graph = Path("server_workspace/data/processed/hprc_r2_sv/full_segments.csv.gz")
    graph_provenance = fingerprint(graph)
    if graph_provenance["sha256"] != canonical["full_segments_sha256"]:
        raise ValueError("Canonical graph changed")
    for fraction in FRACTIONS:
        for job in jobs:
            checkpoint = checkpoint_for(
                args.checkpoint_root / f"fraction_{fraction:g}", job
            )
            validate_checkpoint_holdout(
                checkpoint, test_chrs=set(job.test), closure=job.closure, seed=job.seed
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            arguments = payload["args"].copy()
            if set(arguments["val_chrs"]) != set(job.validation):
                raise ValueError("Checkpoint validation chromosomes differ")
            if Path(arguments["full_segments"]).resolve() != graph.resolve():
                raise ValueError("Checkpoint used a different graph path")
            if arguments["epochs"] != 100 or arguments["patience"] != 20:
                raise ValueError("Incomplete full-duration training protocol")
            manifest = Path(arguments.pop("manifest"))
            arguments.pop("out_dir")
            arg_hash = hashlib.sha256(
                json.dumps(arguments, sort_keys=True).encode()
            ).hexdigest()
            model, predictor = _build_model_from_checkpoint(
                payload,
                _namespace_from_checkpoint(payload, seed=job.seed),
                torch.device("cpu"),
            )
            components = [model] + ([] if predictor is None else [predictor])
            shapes = [
                [(k, list(v.shape)) for k, v in m.named_parameters()]
                for m in components
            ]
            architecture = hashlib.sha256(json.dumps(shapes).encode()).hexdigest()
            parameter_count = sum(p.numel() for m in components for p in m.parameters())
            files = list(
                checkpoint.parent.glob(f"{job.closure}_pooled_predictions*.csv.gz")
            )
            training_files = list(checkpoint.parent.glob("gat_results*.csv"))
            if len(files) != 1 or len(training_files) != 1:
                raise ValueError(
                    "Missing or ambiguous persisted scaling predictions/results"
                )
            frame = pd.read_csv(files[0])
            test = frame.loc[frame.split.eq("heldout_chr_test")]
            validation = frame.loc[frame.split.str.startswith("val_chr_")]
            if test.y_true.nunique() != 2 or not np.isfinite(test.p_edge).all():
                raise ValueError("Invalid held-out reconstruction scores")
            test_hash, validation_hash = target_digest(test), target_digest(validation)
            window_ap = [metric_row(g)["auprc"] for _, g in test.groupby("slice")]
            if not np.isfinite(window_ap).all():
                raise ValueError("Undefined window reconstruction AP")
            training = pd.read_csv(training_files[0])
            active = training.loc[
                training.split.eq("train_chr") & ~training.skipped.astype(bool)
            ]
            meta = dict(
                fraction=fraction, fold=job.fold, seed=job.seed, context=job.closure
            )
            rows.append(
                dict(
                    **meta,
                    **metric_row(test),
                    mean_window_auprc=float(np.mean(window_ap)),
                    n_test_windows=len(window_ap),
                    n_training_windows=len(active),
                    training_candidate_count=int((active.n_pos + active.n_neg).sum()),
                    epochs_run=int(payload["epochs_run"]),
                    epochs_max=100,
                    patience=20,
                    n_parameters=parameter_count,
                    architecture_sha256=architecture,
                    arguments_sha256=arg_hash,
                    test_targets_sha256=test_hash,
                    validation_targets_sha256=validation_hash,
                )
            )
            sources.append(
                dict(
                    **meta,
                    checkpoint=fingerprint(checkpoint),
                    manifest=fingerprint(manifest),
                    predictions=fingerprint(files[0]),
                    training_results=fingerprint(training_files[0]),
                )
            )
            print(f"Audited {len(rows)}/120: {meta}", flush=True)
    frame = pd.DataFrame(rows)
    validate_scales(frame)
    absolute, paired = summaries(frame, args.n_bootstrap, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.out_dir / "per_run.csv", index=False)
    absolute.to_csv(args.out_dir / "summary.csv", index=False)
    paired.to_csv(args.out_dir / "paired_vs_full.csv", index=False)
    figures(absolute, paired, args.out_dir)
    (args.out_dir / "audit.json").write_text(
        json.dumps(
            dict(
                status="complete",
                n_runs=len(frame),
                graph=graph_provenance,
                sources=sources,
                primary_estimand="AP after pooling heldout_chr_test targets within each fold/seed",
                secondary_estimand="Unweighted mean test-window AP within each fold/seed",
                comparison="All four fractions have identical held-out targets, validation targets and arguments except training manifest/output directory",
                uncertainty="Paired hierarchical bootstrap: five chromosome folds, three seeds per fold; overlapping training folds limit independence",
                n_bootstrap=args.n_bootstrap,
                bootstrap_seed=args.seed,
                limitation="Nested window scaling; epochs/early stopping fixed, optimizer updates and training time are not held constant. Not a scaling law or biological result.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

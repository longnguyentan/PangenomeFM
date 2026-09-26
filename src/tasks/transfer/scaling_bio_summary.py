"""Aggregate frozen biological scaling probes after every task is complete."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tasks.entex.analyze import estimate


FEATURE_ALIASES = {
    "coordinate": "C",
    "sequence_kmer": "K",
    "frozen_sequence_fm": "S",
    "frozen_pangenomefm": "T",
    "coordinate_plus_frozen_sequence_fm": "C+S",
    "coordinate_plus_frozen_pangenomefm": "C+T",
    "coordinate_plus_frozen_sequence_fm_plus_frozen_pangenomefm": "C+S+T",
}


def canonical_feature(value: str) -> str:
    return FEATURE_ALIASES.get(value.removesuffix("_pair"), value)


def balanced_accuracy(row: dict) -> float:
    """Recover specificity from accuracy = prevalence*TPR + (1-prevalence)*TNR."""
    prevalence = float(row["positive_fraction"])
    if not 0 < prevalence < 1:
        raise ValueError("Balanced accuracy requires both classes")
    recall = float(row["recall"])
    specificity = (float(row["accuracy"]) - prevalence * recall) / (1 - prevalence)
    if not -1e-9 <= specificity <= 1 + 1e-9:
        raise ValueError("Inconsistent accuracy/prevalence/recall")
    value = (recall + specificity) / 2
    if "balanced_accuracy" in row and not np.isclose(row["balanced_accuracy"], value):
        raise ValueError("Stored balanced accuracy disagrees with confusion rates")
    return value


def collect(
    root: Path, expected_fractions: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0)
) -> pd.DataFrame:
    rows = []
    for fraction in expected_fractions:
        for task in ("sv", "ccre", "ctcf"):
            paths = sorted(
                (root / f"fraction_{fraction:g}" / task).glob(
                    "fold_*/seed_*/*/metrics.csv"
                )
            )
            for path in paths:
                frame = pd.read_csv(path)
                frame = frame.loc[
                    frame.scope.eq("all_test_chromosomes") & frame.chromosome.eq("all")
                ].copy()
                if frame.empty:
                    raise ValueError(f"No all-test metric in {path}")
                keys = frame[["fold", "seed", "closure"]].drop_duplicates()
                if len(keys) != 1 or len(frame) != frame.feature_set.nunique():
                    raise ValueError(f"Duplicate or mixed run in {path}")
                if set(frame.feature_set.map(canonical_feature)) != set(
                    FEATURE_ALIASES.values()
                ):
                    raise ValueError(f"Expected all seven feature sets in {path}")
                for row in frame.to_dict("records"):
                    rows.append(
                        dict(
                            fraction=fraction,
                            task=task,
                            fold=row["fold"],
                            seed=int(row["seed"]),
                            context=row["closure"],
                            feature_set=canonical_feature(str(row["feature_set"])),
                            n_train=int(row["n_train"]),
                            n_val=int(row["n_validation"]),
                            n_test=int(row["n"]),
                            positive_prevalence=float(row["positive_fraction"]),
                            auprc=float(row["auprc"])
                            if pd.notna(row["auprc"])
                            else np.nan,
                            auroc=float(row["auroc"])
                            if pd.notna(row["auroc"])
                            else np.nan,
                            balanced_accuracy=balanced_accuracy(row),
                            f1=float(row["f1"]),
                            precision=float(row["precision"]),
                            recall=float(row["recall"]),
                            source=str(path),
                        )
                    )
    result = pd.DataFrame(rows)
    if result.empty:
        raise FileNotFoundError("No completed scaling biological metrics")
    return result


def validate(
    frame: pd.DataFrame,
    *,
    fractions=(0.125, 0.25, 0.5, 1.0),
    tasks=("sv", "ccre", "ctcf"),
    folds=("fold_a", "fold_b", "fold_c", "fold_d", "fold_e"),
    seeds=(42, 314159, 20260806),
    contexts=("strict", "1hop"),
) -> None:
    keys = ["fraction", "task", "fold", "seed", "context"]
    if frame.duplicated(keys + ["feature_set"]).any():
        raise ValueError("Duplicate biological scaling result")
    expected = set(
        product(fractions, tasks, folds, seeds, contexts, FEATURE_ALIASES.values())
    )
    observed = set(frame[keys + ["feature_set"]].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError(
            f"Incomplete or unexpected biological scaling matrix: "
            f"{len(expected - observed)} missing, {len(observed - expected)} unexpected"
        )
    for key, group in frame.groupby(keys):
        if group.n_test.nunique() != 1 or group.positive_prevalence.nunique() != 1:
            raise ValueError(f"Paired task universe changed in {key}")
        if set(group.feature_set) != set(FEATURE_ALIASES.values()):
            raise ValueError(f"Missing feature set in {key}")
    for key, group in frame.groupby(["task", "fold", "seed", "context", "feature_set"]):
        for column in ["n_test", "positive_prevalence", "n_train", "n_val"]:
            if group[column].nunique() != 1:
                raise ValueError(f"Cross-fraction universe changed: {key}, {column}")


def summarize(
    frame: pd.DataFrame, n_bootstrap: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    absolute, paired = [], []
    for (fraction, task, context), group in frame.groupby(
        ["fraction", "task", "context"]
    ):
        for feature, values in group.groupby("feature_set"):
            for metric in [
                "auprc",
                "auroc",
                "balanced_accuracy",
                "f1",
                "precision",
                "recall",
                "positive_prevalence",
                "n_train",
                "n_val",
                "n_test",
            ]:
                absolute.append(
                    dict(
                        fraction=fraction,
                        task=task,
                        context=context,
                        feature_set=feature,
                        metric=metric,
                        **estimate(values, metric, n_bootstrap, seed),
                    )
                )
        for baseline in ["C+S", "C+T"]:
            wide = group.pivot(
                index=["fold", "seed"], columns="feature_set", values="auprc"
            )
            if baseline not in wide or "C+S+T" not in wide:
                raise ValueError("Missing paired biological comparator")
            gain = (wide["C+S+T"] - wide[baseline]).rename("gain").reset_index()
            paired.append(
                dict(
                    fraction=fraction,
                    task=task,
                    context=context,
                    contrast=f"C+S+T minus {baseline}",
                    metric="auprc",
                    **estimate(gain, "gain", n_bootstrap, seed),
                )
            )
    return pd.DataFrame(absolute), pd.DataFrame(paired)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument(
        "--sv-examples",
        type=Path,
        default=Path(
            "server_workspace/data/processed/hgsvc3_sv_breakpoint_examples_20260809/sv_breakpoint_examples.csv.gz"
        ),
    )
    ap.add_argument(
        "--complexity",
        type=Path,
        default=Path(
            "server_workspace/results/complexity_context_v2_20260815/native_complexity_v2/complexity_features.tsv"
        ),
    )
    ap.add_argument(
        "--manuscript-config",
        type=Path,
        default=Path("configs/server_full_multicohort_20260806.json"),
    )
    args = ap.parse_args()
    frame = collect(args.probe_root)
    config = json.loads(args.manuscript_config.read_text())
    validate(
        frame,
        folds=tuple(f["name"] for f in config["rotating_chromosome_folds"]),
        seeds=tuple(config["training"]["seeds"]),
        contexts=tuple(config["training"]["contexts"]),
    )
    from tasks.transfer.scaling_prediction_audit import audit_predictions

    prediction_audit, baseline_audit = audit_predictions(frame)
    absolute, paired = summarize(frame, args.n_bootstrap, 20260924)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.out_dir / "per_run.csv", index=False)
    absolute.to_csv(args.out_dir / "summary.csv", index=False)
    paired.to_csv(args.out_dir / "paired_gains.csv", index=False)
    prediction_audit.to_csv(args.out_dir / "prediction_audit.csv", index=False)
    baseline_audit.to_csv(args.out_dir / "baseline_invariance.csv", index=False)
    from tasks.transfer.scaling_bio_report import write_report

    write_report(
        frame,
        args.out_dir,
        args.sv_examples,
        args.complexity,
        args.n_bootstrap,
        20260924,
    )
    (args.out_dir / "audit.json").write_text(
        json.dumps(
            dict(
                status="complete",
                expected_tasks=["sv", "ccre", "ctcf"],
                expected_fractions=[0.125, 0.25, 0.5, 1.0],
                expected_runs_per_task_fraction=30,
                encoders_frozen=True,
                primary_comparison="AUPRC(C+S+T)-AUPRC(C+S)",
                paired_comparisons_use_same_fold_seed_test_universe=True,
                uncertainty="Hierarchical fold/seed bootstrap; overlapping training folds limit independence",
                n_bootstrap=args.n_bootstrap,
                seed=20260924,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

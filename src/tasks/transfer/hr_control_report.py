"""Audit actual paired predictions before summarizing frozen H/R controls."""

from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from evaluation.calibration import EPSILON
from evaluation.paired_inference import bh_adjust, fold_sign_flip
from scripts.server.run_ccre_frozen_probe_fold import binary_metrics
from scripts.summarize_v2_review_controls import _feature_names
from tasks.entex.analyze import estimate
from tasks.entex.prepare import fingerprint
from tasks.transfer.scaling_prediction_audit import target_digest


def audited_run(directory: Path, model: str, task: str, fold: str, seed: int,
                context: str, *, validation_only: bool = False) -> pd.DataFrame:
    audit = json.loads((directory / "audit.json").read_text())
    if (audit.get("status") != "complete" or audit.get("fold") != fold
            or audit.get("seed") != seed or audit.get("closure") != context
            or not audit.get("checkpoint_sha256")):
        raise ValueError(f"Incomplete or mismatched run: {directory}")
    partition = "development_validation" if validation_only else "test"
    if audit.get("evaluation_partition", "test") != partition:
        raise ValueError("Requested and recorded evaluation partitions differ")
    if validation_only and (audit.get("heldout_predictions_produced", True)
                            or (directory / "test_predictions.csv.gz").exists()):
        raise ValueError("Development run unexpectedly produced held-out predictions")
    metrics = pd.read_csv(directory / "metrics.csv", float_precision="round_trip")
    if metrics.feature_set.duplicated().any():
        raise ValueError("Duplicate feature metrics")
    names = _feature_names(task)
    metrics = metrics.set_index("feature_set")
    identity = "example_id" if task == "sv" else "segid"
    columns = [identity, "chromosome", "y_true", "p_calibrated", "feature_set", "threshold", "y_pred"]
    prediction_file = "validation_predictions.csv.gz" if validation_only else "test_predictions.csv.gz"
    predictions = pd.read_csv(directory / prediction_file, usecols=columns,
                              float_precision="round_trip")
    rows = []
    for feature, raw in names.items():
        if raw not in metrics.index:
            raise ValueError(f"Missing comparator {raw}")
        metric = metrics.loc[raw]
        if (metric["fold"] != fold or int(metric["seed"]) != seed or metric["closure"] != context):
            raise ValueError("Stored metric partition mismatch")
        p = predictions.loc[predictions.feature_set.eq(raw)].copy()
        digest = target_digest(p, identity, "chromosome")
        count = metric.n_validation if validation_only else metric.n_test
        chrom_key = "validation_chromosomes" if validation_only else "test_chromosomes"
        if validation_only and (metric.n_test != 0 or metric.scope != partition):
            raise ValueError("Development metric is mislabeled as a test result")
        if (len(p) != count or p.y_true.nunique() != 2
                or set(p.chromosome) != set(audit[chrom_key])
                or not p.threshold.eq(float(metric.threshold)).all()
                or not p.p_calibrated.between(0, 1).all()):
            raise ValueError(f"Prediction universe, threshold or scores invalid: {directory}/{raw}")
        replay = binary_metrics(p.y_true.to_numpy(), p.p_calibrated.to_numpy(), float(metric.threshold))
        for name in ["auprc", "auroc", "f1", "precision", "recall", "positive_fraction"]:
            if not np.isclose(replay[name], metric[name], rtol=0, atol=1e-10):
                raise ValueError(f"Stored {name} differs from actual predictions")
        # Match the manuscript's explicit clipping convention at saturated scores.
        probability = np.clip(p.p_calibrated.to_numpy(), EPSILON, 1 - EPSILON)
        predicted = probability >= float(metric.threshold)
        prevalence = replay["positive_fraction"]
        ordered = p.assign(_id=p[identity].astype(str)).sort_values("_id")
        score_digest = hashlib.sha256(ordered.p_calibrated.to_numpy("<f8").tobytes()).hexdigest()
        rows.append(dict(model=model, task=task, fold=fold, seed=seed, context=context,
                         feature=feature, embedding="R" if model.endswith("random") else "T",
                         evaluation_partition=partition, n_evaluated=len(p),
                         n_train=int(metric.n_train), n_val=int(metric.n_validation),
                         n_test=0 if validation_only else len(p),
                         positive_prevalence=prevalence,
                         auprc=replay["auprc"], auroc=replay["auroc"],
                         normalized_ap=(replay["auprc"] - prevalence) / (1 - prevalence),
                         balanced_accuracy=float(balanced_accuracy_score(p.y_true, predicted)),
                         f1=replay["f1"], precision=replay["precision"], recall=replay["recall"],
                         targets_sha256=digest, scores_sha256=score_digest,
                         checkpoint_sha256=audit["checkpoint_sha256"], source=str(directory)))
    return pd.DataFrame(rows)


def validate_pairs(frame: pd.DataFrame, folds: list[str], seeds: list[int]) -> None:
    keys = ["fold", "seed", "context", "task", "model", "feature"]
    expected = set(product(folds, seeds, ["strict", "1hop"], ["sv", "ccre"],
                           ["v1", "random"], ["cs", "csh", "cst", "csht"]))
    if frame.duplicated(keys).any() or set(frame[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("Incomplete, duplicate or unexpected H/R comparison matrix")
    for key, group in frame.groupby(["fold", "seed", "context", "task"]):
        for field in ["targets_sha256", "n_train", "n_val", "n_test", "positive_prevalence"]:
            if group[field].nunique() != 1:
                raise ValueError(f"Paired example/label universe mismatch: {key}/{field}")
        for feature in ["cs", "csh"]:
            if group.loc[group.feature.eq(feature), "scores_sha256"].nunique() != 1:
                raise ValueError(f"Non-embedding baseline predictions changed: {key}/{feature}")


def summarize(frame: pd.DataFrame, n_bootstrap: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    absolute, contrasts = [], []
    for (task, context, model, feature), group in frame.groupby(["task", "context", "model", "feature"]):
        for metric in ["auprc", "auroc", "normalized_ap", "balanced_accuracy", "f1"]:
            absolute.append(dict(task=task, context=context, model=model, feature=feature, metric=metric,
                                 **estimate(group, metric, n_bootstrap, 20260927)))
    for (task, context), group in frame.groupby(["task", "context"]):
        for metric in ["auprc", "auroc", "normalized_ap"]:
            wide = group.pivot(index=["fold", "seed"], columns=["model", "feature"], values=metric)
            comparisons = [
                ("T_given_CS", ("v1", "cst"), ("v1", "cs")),
                ("T_given_CSH", ("v1", "csht"), ("v1", "csh")),
                ("R_given_CS", ("random", "cst"), ("random", "cs")),
                ("R_given_CSH", ("random", "csht"), ("random", "csh")),
                ("T_minus_R_given_CS", ("v1", "cst"), ("random", "cst")),
                ("T_minus_R_given_CSH", ("v1", "csht"), ("random", "csht")),
            ]
            for name, left, right in comparisons:
                values = (wide[left] - wide[right]).rename("gain").reset_index()
                contrasts.append(dict(task=task, context=context, metric=metric, contrast=name,
                                      sign_flip_p=fold_sign_flip(values),
                                      **estimate(values, "gain", n_bootstrap, 20260927)))
    contrasts = pd.DataFrame(contrasts)
    contrasts["bh_q_within_metric"] = contrasts.groupby("metric").sign_flip_p.transform(bh_adjust)
    return pd.DataFrame(absolute), contrasts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--campaign-root", type=Path)
    group.add_argument("--single-root", type=Path, help="Original fold-A/42 controls root; audit only this declared scope")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--wait-hours", type=float, default=0,
                        help="Bounded dependency wait for the already launched campaign; never retries or fits probes")
    args = parser.parse_args()
    config = json.loads(Path("configs/server_full_multicohort_20260806.json").read_text())
    folds = ["fold_a"] if args.single_root else [fold["name"] for fold in config["rotating_chromosome_folds"]]
    seeds = [42] if args.single_root else config["training"]["seeds"]
    status_path = args.out_dir.parent / (args.out_dir.name + "_status.json")

    def write_status(state: str, **extra) -> None:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(status=state, **extra), indent=2) + "\n")
        temporary.replace(status_path)

    if args.wait_hours:
        if not args.campaign_root or args.wait_hours < 0:
            parser.error("A positive wait requires --campaign-root")
        started = time.monotonic()
        expected = [args.campaign_root / fold / f"seed_{seed}" / "status.json"
                    for fold, seed in product(folds, seeds)]
        while True:
            states = []
            for path in expected:
                if path.exists():
                    try:
                        states.append(json.loads(path.read_text()))
                    except json.JSONDecodeError:
                        # The running legacy writer may be replacing its receipt.
                        # A partial write is never treated as completion.
                        states.append({})
            if any(s.get("status") == "failed" for s in states):
                write_status("failed", error="An upstream control run failed")
                raise RuntimeError("An upstream control run failed; no summary created")
            complete = sum(s.get("status") == "complete" for s in states)
            if complete == len(expected):
                break
            if time.monotonic() - started > args.wait_hours * 3600:
                write_status("failed", error="Timed out waiting for the fixed control matrix", completed=complete)
                raise TimeoutError("Incomplete H/R campaign; no summary created")
            write_status("waiting_for_campaign", completed_fold_seed_jobs=complete, expected_fold_seed_jobs=len(expected))
            time.sleep(30)
    write_status("auditing_predictions")
    try:
        frames, receipts = [], []
        for fold, seed in product(folds, seeds):
            root = args.single_root or args.campaign_root / fold / f"seed_{seed}"
            status = json.loads((root / "status.json").read_text())
            if (status.get("status") != "complete" or status["completed_commands"] != len(status["commands"])
                    or status["fold"]["name"] != fold or status["seed"] != seed
                    or status["random_initialization_seed"] != seed or not status["encoders_frozen"]):
                raise ValueError(f"Incomplete or mismatched control receipt: {root}")
            receipts.append(fingerprint(root / "status.json"))
            for model, task, context in product(["v1", "random"], ["sv", "ccre"], ["strict", "1hop"]):
                frames.append(audited_run(root / "probes" / model / task / context,
                                          model, task, fold, seed, context))
        frame = pd.concat(frames, ignore_index=True)
        validate_pairs(frame, folds, seeds)
        absolute, paired = summarize(frame, args.n_bootstrap)
        args.out_dir.mkdir(parents=True, exist_ok=False)
        frame.to_csv(args.out_dir / "audited_per_run.csv", index=False)
        absolute.to_csv(args.out_dir / "summary.csv", index=False)
        paired.to_csv(args.out_dir / "paired_gains.csv", index=False)
        (args.out_dir / "audit.json").write_text(json.dumps(dict(
            status="complete", n_runs=len(frame) // 4, n_folds=len(folds), seeds=seeds,
            source_receipts=receipts, prediction_identity_and_baseline_invariance="passed",
            scope="exploratory previously inspected chromosomes; no v2 or external confirmation",
            n_bootstrap=args.n_bootstrap, bootstrap_seed=20260927,
        ), indent=2) + "\n")
        write_status("complete", summary=str(args.out_dir), n_runs=len(frame) // 4)
    except Exception as exc:
        write_status("failed", error=f"{type(exc).__name__}: {exc}")
        raise



if __name__ == "__main__":
    main()

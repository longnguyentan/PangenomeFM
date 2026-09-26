"""Audit actual biological scaling prediction identities, labels and metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from evaluation.calibration import EPSILON

from tasks.transfer.scaling_bio_summary import canonical_feature


def target_digest(frame: pd.DataFrame, identity: str, chromosome: str) -> str:
    columns = [identity, chromosome, "y_true"]
    if frame[columns].isna().any().any() or frame[identity].duplicated().any():
        raise ValueError("Missing or duplicate prediction identity")
    values = frame[columns].astype(str).sort_values(identity).to_csv(index=False)
    return hashlib.sha256(values.encode()).hexdigest()


def audit_predictions(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for source, expected in metrics.groupby("source", sort=True):
        directory = Path(source).parent
        audit = json.loads((directory / "audit.json").read_text())
        if audit.get("status") != "complete":
            raise ValueError(f"Incomplete run audit: {directory}")
        checkpoint = audit.get("checkpoint_sha256") or audit.get("checkpoint", {}).get(
            "sha256"
        )
        if not checkpoint:
            raise ValueError(f"Missing checkpoint provenance: {directory}")
        parquet = directory / "predictions.parquet"
        if parquet.exists():
            identity, chromosome = "measurement_id", "chrom"
            predictions = pd.read_parquet(
                parquet,
                columns=[
                    identity,
                    chromosome,
                    "y_true",
                    "feature_set",
                    "p_calibrated",
                    "y_pred",
                ],
            )
            path = parquet
        else:
            path = directory / "test_predictions.csv.gz"
            columns = pd.read_csv(path, nrows=0).columns
            identity = "example_id" if "example_id" in columns else "segid"
            chromosome = "chromosome"
            predictions = pd.read_csv(
                path,
                usecols=[
                    identity,
                    chromosome,
                    "y_true",
                    "feature_set",
                    "p_calibrated",
                    "y_pred",
                ],
            )
        predictions["feature_set"] = predictions.feature_set.map(canonical_feature)
        for metric in expected.to_dict("records"):
            p = predictions.loc[
                predictions.feature_set.eq(metric["feature_set"])
            ].copy()
            if len(p) != metric["n_test"] or not np.isclose(
                p.y_true.mean(), metric["positive_prevalence"]
            ):
                raise ValueError(f"Prediction/metric universe mismatch: {source}")
            if not np.isfinite(p.p_calibrated).all():
                raise ValueError(f"Nonfinite scores: {source}")
            for key, value in {
                "auprc": average_precision_score(
                    p.y_true, np.clip(p.p_calibrated, EPSILON, 1 - EPSILON)
                ),
                "balanced_accuracy": balanced_accuracy_score(p.y_true, p.y_pred),
            }.items():
                if not np.isclose(value, metric[key], atol=1e-7, rtol=0):
                    raise ValueError(f"Stored {key} differs from predictions: {source}")
            digest = target_digest(p, identity, chromosome)
            scores = (
                p.assign(_id=p[identity].astype(str))
                .sort_values("_id")
                .p_calibrated.to_numpy("<f8")
            )
            rows.append(
                dict(
                    **{
                        k: metric[k]
                        for k in [
                            "fraction",
                            "task",
                            "fold",
                            "seed",
                            "context",
                            "feature_set",
                        ]
                    },
                    targets_sha256=digest,
                    scores_sha256=hashlib.sha256(scores.tobytes()).hexdigest(),
                    checkpoint_sha256=checkpoint,
                    prediction_path=str(path),
                    prediction_bytes=path.stat().st_size,
                    prediction_mtime_ns=path.stat().st_mtime_ns,
                    auprc=metric["auprc"],
                )
            )
    frame = pd.DataFrame(rows)
    for key, group in frame.groupby(["task", "fold", "seed", "context"]):
        if group.targets_sha256.nunique() != 1:
            raise ValueError(
                f"Actual test identities/labels changed across features or fractions: {key}"
            )
    controls = []
    for key, group in frame.loc[frame.feature_set.isin(["C", "K", "S", "C+S"])].groupby(
        ["task", "fold", "seed", "context", "feature_set"]
    ):
        controls.append(
            dict(
                zip(["task", "fold", "seed", "context", "feature_set"], key),
                exact_score_invariance=group.scores_sha256.nunique() == 1,
                ap_range=float(group.auprc.max() - group.auprc.min()),
            )
        )
    return frame, pd.DataFrame(controls)

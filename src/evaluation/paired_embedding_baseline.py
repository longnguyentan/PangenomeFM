"""Same-example endpoint-pair classifier for frozen genomic representations.

This is the adapter used for sequence foundation models such as DeepGene or
Nucleotide Transformer.  It does not claim to reproduce their published task:
it freezes their endpoint profiles and trains the same small pair classifier on
the canonical PangenomeFM labels and splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_embedding_lookup(path: Path) -> tuple[dict[str, int], np.ndarray]:
    packed = np.load(path, allow_pickle=False)
    if not {"ids", "embeddings"}.issubset(packed.files):
        raise ValueError("Embedding NPZ requires arrays named ids and embeddings")
    ids = packed["ids"].astype(str)
    embeddings = packed["embeddings"].astype(np.float32)
    if embeddings.ndim != 2 or len(ids) != len(embeddings):
        raise ValueError("Embeddings must have shape [len(ids), dimension]")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Embedding IDs are not unique")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain non-finite values")
    return {value: index for index, value in enumerate(ids)}, embeddings


def pair_features(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Symmetric features so endpoint order cannot change a prediction."""

    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Endpoint embeddings require matching [examples, dim] arrays")
    return np.concatenate([left * right, np.abs(left - right)], axis=1)


def evaluate(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    metrics = {
        "n_examples": int(len(labels)),
        "positive_count": int((labels == 1).sum()),
        "negative_count": int((labels == 0).sum()),
    }
    if len(np.unique(labels)) == 2:
        metrics.update(
            {
                "auroc": float(roc_auc_score(labels, probabilities)),
                "auprc": float(average_precision_score(labels, probabilities)),
                "negative_log_likelihood": float(
                    log_loss(labels, probabilities, labels=[0, 1])
                ),
                "brier": float(np.mean((probabilities - labels) ** 2)),
            }
        )
    return metrics


def run(
    *,
    manifest_path: Path,
    embeddings_path: Path,
    out_dir: Path,
    method_name: str,
    left_id_column: str = "source_segment_name",
    right_id_column: str = "destination_segment_name",
    split_column: str = "candidate_partition",
    require_complete: bool = False,
    seed: int = 20260823,
) -> dict[str, Any]:
    manifest = pd.read_parquet(manifest_path) if manifest_path.suffix == ".parquet" else pd.read_csv(manifest_path)
    required = {"example_id", left_id_column, right_id_column, "label", split_column}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"Benchmark manifest is missing columns: {sorted(missing)}")
    if manifest["example_id"].duplicated().any():
        raise ValueError("Benchmark example IDs are not unique")
    labels_all = pd.to_numeric(manifest["label"], errors="coerce")
    if not labels_all.isin([0, 1]).all():
        raise ValueError("Benchmark labels must be binary")
    allowed_splits = {"train", "validation", "test"}
    if not set(manifest[split_column].astype(str)).issubset(allowed_splits):
        raise ValueError(f"{split_column} must contain train/validation/test")

    lookup, embeddings = load_embedding_lookup(embeddings_path)
    left_ids = manifest[left_id_column].astype(str)
    right_ids = manifest[right_id_column].astype(str)
    converted = left_ids.isin(lookup) & right_ids.isin(lookup)
    failures = manifest.loc[~converted, ["example_id", left_id_column, right_id_column]].copy()
    failures["reason"] = np.where(
        ~left_ids[~converted].isin(lookup) & ~right_ids[~converted].isin(lookup),
        "both_endpoint_embeddings_missing",
        np.where(~left_ids[~converted].isin(lookup), "source_embedding_missing", "destination_embedding_missing"),
    )
    if require_complete and not converted.all():
        raise ValueError(f"Only {int(converted.sum())}/{len(manifest)} examples have both endpoint embeddings")
    work = manifest.loc[converted].copy().reset_index(drop=True)
    if work.empty:
        raise ValueError("No examples could be converted")
    left = np.stack([embeddings[lookup[value]] for value in work[left_id_column].astype(str)])
    right = np.stack([embeddings[lookup[value]] for value in work[right_id_column].astype(str)])
    features = pair_features(left, right)
    labels = work["label"].to_numpy(np.int64)
    splits = work[split_column].astype(str).to_numpy()
    for name in allowed_splits:
        if not np.any(splits == name):
            raise ValueError(f"Converted examples contain no {name} rows")
    if len(np.unique(labels[splits == "train"])) != 2:
        raise ValueError("Training examples require both classes")

    candidates = [0.01, 0.1, 1.0, 10.0]
    selected_c = candidates[0]
    best_validation = -np.inf
    for value in candidates:
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=value,
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
            ),
        )
        estimator.fit(features[splits == "train"], labels[splits == "train"])
        validation_probability = estimator.predict_proba(features[splits == "validation"])[:, 1]
        if len(np.unique(labels[splits == "validation"])) == 2:
            score = average_precision_score(labels[splits == "validation"], validation_probability)
        else:
            score = -log_loss(labels[splits == "validation"], validation_probability, labels=[0, 1])
        if score > best_validation:
            best_validation = float(score)
            selected_c = value

    fit_mask = np.isin(splits, ["train", "validation"])
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=selected_c,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        ),
    )
    estimator.fit(features[fit_mask], labels[fit_mask])
    probability = estimator.predict_proba(features)[:, 1]
    work["method"] = method_name
    work["probability"] = probability

    split_metrics = {
        name: evaluate(labels[splits == name], probability[splits == name])
        for name in ["train", "validation", "test"]
    }
    test_frame = work.loc[work[split_column].eq("test")]
    strata: list[dict[str, Any]] = []
    for columns in [["context"], ["chromosome"], ["context", "locus_complexity_category"]]:
        if not set(columns).issubset(test_frame):
            continue
        grouper: str | list[str] = columns[0] if len(columns) == 1 else columns
        for key, group in test_frame.groupby(grouper, dropna=False):
            keys = (key,) if len(columns) == 1 else key
            strata.append(
                {
                    **dict(zip(columns, keys)),
                    **evaluate(group["label"].to_numpy(), group["probability"].to_numpy()),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    work.to_csv(out_dir / "predictions.csv.gz", index=False, compression="gzip")
    failures.to_csv(out_dir / "conversion_failures.tsv", sep="\t", index=False)
    pd.DataFrame(strata).to_csv(out_dir / "test_strata.tsv", sep="\t", index=False)
    summary = {
        "status": "complete",
        "method": method_name,
        "comparison_tier": "approximately_matched_frozen_embedding_adapter",
        "published_task_reproduction": False,
        "requested_examples": int(len(manifest)),
        "converted_examples": int(converted.sum()),
        "conversion_success_fraction": float(converted.mean()),
        "selected_logistic_c": selected_c,
        "selection_split": "validation",
        "fit_after_selection": "train_plus_validation",
        "split_metrics": split_metrics,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "embeddings": str(embeddings_path.resolve()),
        "embeddings_sha256": sha256_file(embeddings_path),
        "seed": seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--left-id-column", default="source_segment_name")
    parser.add_argument("--right-id-column", default="destination_segment_name")
    parser.add_argument("--split-column", default="candidate_partition")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    run(
        manifest_path=args.manifest,
        embeddings_path=args.embeddings,
        out_dir=args.out_dir,
        method_name=args.method_name,
        left_id_column=args.left_id_column,
        right_id_column=args.right_id_column,
        split_column=args.split_column,
        require_complete=args.require_complete,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""Audit native v2 candidate coverage and validation-only nuisance baselines.

This calls the trainer's loader, complete-group packing, query masking,
DropEdge and visible-degree functions. It does not train an encoder, change
candidates, or score held-out chromosomes. A fixed per-slice random draw is a
protocol replay, not the historical trainer's exact stochastic trace.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import subprocess
import zlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from evaluation.external import _namespace_from_checkpoint
from evaluation.splits import normalize_chrom, validate_chromosome_split
from graph.features import build_oid_metadata_from_segments
from graph.junction_repair import group_batches
from graph.slicing import build_global_index
from tasks.entex.prepare import fingerprint
from training.node_inputs import visible_structure_features
from training.pretrain import (
    drop_edges, leakage_safe_mask_indices, load_slice,
    mask_positive_query_edges, tensorize_slice,
)

GEOMETRY = ["signed_offset", "signed_gap", "same_orientation", "same_sn",
            "absolute_offset", "absolute_gap"]
DEGREE = ["degree_u", "degree_v", "degree_sum", "degree_product", "degree_difference"]


def validate_groups(sd: dict) -> None:
    """Check balance after the native loader's node filtering, not before it."""
    memberships = np.full(len(sd["labels"]), "", dtype=object)
    for split in ["train", "val", "test"]:
        memberships[sd[split + "_idx"]] = split
    if (memberships == "").any():
        raise ValueError("Candidate without an internal split")
    for group in np.unique(sd["group_ids"]):
        idx = np.flatnonzero(sd["group_ids"] == group)
        positive = idx[sd["labels"][idx] == 1]
        negative = idx[sd["labels"][idx] == 0]
        if len(positive) != len(negative) or not len(positive) or len(set(memberships[idx])) != 1:
            raise ValueError("Unbalanced or split junction group after native filtering")
        for endpoint in ["query_u", "query_v"]:
            if Counter(sd[endpoint][positive]) != Counter(sd[endpoint][negative]):
                raise ValueError("Endpoint imbalance after native filtering")


def feature_rows(sd: dict, args: argparse.Namespace, partition: str) -> pd.DataFrame:
    """One fixed packed training pass; validation uses the native evaluation mask."""
    if partition not in {"train", "validation"}:
        raise ValueError("Held-out chromosome features are deliberately not scored")
    validate_groups(sd)
    native = tensorize_slice(sd, torch.device("cpu"), args)
    seed = (zlib.crc32(sd["name"].encode()) ^ int(args.seed)) & 0x7FFFFFFF
    if partition == "train":
        batches = group_batches(sd["train_groups"], args.batch_size, np.random.default_rng(seed))
    else:
        batches = [sd["val_idx"]] if len(sd["val_idx"]) >= 4 else []
    frames = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for batch in batches:
            idx = torch.as_tensor(batch, dtype=torch.long)
            hidden = leakage_safe_mask_indices(native, idx)
            src, dst, edge_attr = mask_positive_query_edges(
                native["src"], native["dst"], native["edge_attr"], native["q_u"],
                native["q_v"], native["labels"], hidden, native["node_oids"],
            )
            src, dst, _ = drop_edges(src, dst, args.drop_edge_rate if args.drop_edge else 0,
                                     edge_attr, training=partition == "train")
            x = visible_structure_features(native["X"], src, dst, native["deg_norm"])
            du, dv = x[native["q_u"][idx], 4].numpy(), x[native["q_v"][idx], 4].numpy()
            geom = sd["pair_geom"][batch]
            values = np.column_stack([geom, abs(geom[:, 0]), abs(geom[:, 1]),
                                      du, dv, du + dv, du * dv, abs(du - dv)])
            frame = pd.DataFrame(values, columns=GEOMETRY + DEGREE)
            frame["label"] = sd["labels"][batch].astype(int)
            frame["slice"] = sd["name"]
            frame["chrom"] = normalize_chrom(sd["target_sn"])
            frame["context"] = sd["closure"]
            frame["partition"] = partition
            frame["candidate_index"] = batch
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fit_validation_baselines(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixed linear probes; no direction, regularization or threshold selection."""
    rows, predictions = [], []
    if not set(frame.partition) <= {"train", "validation"}:
        raise ValueError("Baseline fit must not include held-out chromosomes")
    for context, group in frame.groupby("context"):
        train = group.loc[group.partition.eq("train")]
        val = group.loc[group.partition.eq("validation")]
        if set(train.chrom) & set(val.chrom):
            raise ValueError("Training/validation chromosomes overlap")
        if train.label.nunique() != 2 or val.label.nunique() != 2:
            raise ValueError("Both baseline partitions require both classes")
        for name, columns in [("geometry", GEOMETRY), ("visible_degree", DEGREE),
                              ("geometry_and_visible_degree", GEOMETRY + DEGREE)]:
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=42))
            model.fit(train[columns], train.label)
            if model[-1].n_iter_.max() >= model[-1].max_iter:
                raise RuntimeError("Nuisance baseline did not converge")
            p = model.predict_proba(val[columns])[:, 1]
            pred = val[["slice", "chrom", "context", "candidate_index", "label"]].copy()
            pred["baseline"], pred["probability"] = name, p
            predictions.append(pred)
            per_slice = [roc_auc_score(g.label, g.probability) for _, g in pred.groupby("slice")
                         if g.label.nunique() == 2]
            rows.append(dict(context=context, baseline=name, n_train=len(train), n_val=len(val),
                             n_val_slices=len(per_slice), positive_prevalence=val.label.mean(),
                             auprc=average_precision_score(val.label, p), auroc=roc_auc_score(val.label, p),
                             macro_slice_auroc=float(np.mean(per_slice)),
                             scope="pretraining validation chromosomes/internal val candidates only"))
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=False)
    progress = args.out_dir / "audit.json"
    receipt = dict(status="running", checkpoint=fingerprint(args.checkpoint),
                   manifest=fingerprint(args.manifest), full_segments=fingerprint(args.full_segments),
                   code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    progress.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        expected = json.loads(Path("configs/entex_v1.json").read_text())["full_segments_sha256"]
        if receipt["full_segments"]["sha256"] != expected:
            raise ValueError("Graph differs from the exact manuscript HPRC R2 graph")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        native_args = _namespace_from_checkpoint(checkpoint, seed=42)
        if (native_args.objective != "junction_repair" or not native_args.mask_query_edges
                or native_args.node_structure_source != "visible"):
            raise ValueError("A native masked-junction/visible-degree checkpoint configuration is required")
        if args.training_drop_edge_rate is not None:
            if not 0 <= args.training_drop_edge_rate < 1:
                raise ValueError("Training DropEdge rate must be in [0, 1)")
            native_args.drop_edge_rate = args.training_drop_edge_rate
            native_args.drop_edge = args.training_drop_edge_rate > 0
        receipt["training_drop_edge_rate_override"] = args.training_drop_edge_rate
        # Geometry is collected for the control, without changing any encoder.
        native_args.pair_geometry = True
        split = validate_chromosome_split(val_chrs=native_args.val_chrs, test_chrs=native_args.test_chrs)
        receipt.update(native_args=vars(native_args), split=split, checkpoint_weights_used=False,
                       baseline="Fixed standardized logistic C=1; train only; no test scores or hyperparameter search",
                       mask="Native held-out masks; complete packed train groups and one fixed DropEdge draw per slice",
                       limitations="Validation diagnostics, not independent biological performance or a proof of no shortcuts")
        segments = pd.read_csv(args.full_segments, usecols=["id", "name", "LN", "SN", "SO", "SR"])
        seg_index, canonical = build_global_index(segments)
        if len(canonical) != len(segments):
            raise ValueError("Duplicate source segment IDs break native metadata alignment")
        md = build_oid_metadata_from_segments(segments, seg_index)
        manifest = pd.read_csv(args.manifest)
        required = json.loads(Path("configs/server_full_multicohort_20260806.json").read_text())["primary_chromosomes"]
        counts, frames = [], []
        for _, row in manifest.iterrows():
            record = dict(slice=row["name"], chrom=normalize_chrom(row.target_sn), context=row.closure)
            for kind in ["segments", "links"]:
                record[kind + "_sha256"] = fingerprint(Path(row[kind + "_path"]))["sha256"]
            sd = load_slice(row, seg_index, md, segments, native_args, audit_out=record)
            counts.append(record)
            if sd is not None:
                try:
                    validate_groups(sd)
                except ValueError as exc:
                    pd.DataFrame(counts).to_csv(args.out_dir / "per_window.csv", index=False)
                    raise ValueError(f"{row['name']}: {exc}") from exc
                if record["chrom"] not in split["test_chrs"]:
                    partition = "validation" if record["chrom"] in split["val_chrs"] else "train"
                    feature = feature_rows(sd, native_args, partition)
                    if not feature.empty:
                        frames.append(feature)
            if len(counts) % 25 == 0:
                logging.info("Audited %d/%d windows", len(counts), len(manifest))
                progress.write_text(json.dumps(dict(receipt, completed_windows=len(counts)), indent=2) + "\n")
        counts = pd.DataFrame(counts)
        counts.to_csv(args.out_dir / "per_window.csv", index=False)
        counts.groupby(["context", "chrom", "exclusion"]).size().rename("n_windows").to_csv(
            args.out_dir / "coverage.csv")
        missing = {context: sorted(set(required) - set(group.loc[group.exclusion.eq("retained"), "chrom"]))
                   for context, group in counts.groupby("context")}
        if set(counts.context) != {"strict", "1hop"} or any(missing.values()):
            raise ValueError(f"Full canonical coverage gate failed: {missing}")
        for internal in ["val", "test"]:
            eligible = counts.exclusion.eq("retained") & counts[f"n_{internal}_candidates"].ge(4)
            for context in ["strict", "1hop"]:
                absent = set(required) - set(counts.loc[eligible & counts.context.eq(context), "chrom"])
                if absent:
                    raise ValueError(f"No n>=4 native {internal} window for {context}: {sorted(absent)}")
        frame = pd.concat(frames, ignore_index=True)
        frame.to_parquet(args.out_dir / "validation_baseline_features.parquet", index=False)
        scores, predictions = fit_validation_baselines(frame)
        scores.to_csv(args.out_dir / "validation_baselines.csv", index=False)
        predictions.to_parquet(args.out_dir / "validation_predictions.parquet", index=False)
        receipt.update(status="complete", n_windows=len(counts), n_retained=int(counts.exclusion.eq("retained").sum()),
                       missing_chromosomes=missing, endpoint_balance_after_native_filter="passed",
                       heldout_chromosome_predictions_produced=False)
        progress.write_text(json.dumps(receipt, indent=2) + "\n")
    except Exception as exc:
        progress.write_text(json.dumps(dict(receipt, status="failed", error=f"{type(exc).__name__}: {exc}"), indent=2) + "\n")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["checkpoint", "manifest", "full-segments", "out-dir"]:
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--training-drop-edge-rate", type=float,
                        help="Explicit proposed training rate; omit to use the smoke checkpoint setting")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

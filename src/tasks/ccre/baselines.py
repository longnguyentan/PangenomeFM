"""
Logistic-regression / MLP floor baseline for cCRE node classification.

Consumes node_labels.csv.gz from tasks.ccre.label_nodes and the same
structural features the GAT will use (from models.gat),
minus the `is_grch38` shortcut. Trains a multinomial LR and reports:
  - overall macro-F1
  - per-class F1 / precision / recall
  - per-chromosome macro-F1

This is the number the graph model MUST beat to claim anything.

Usage
-----
python -m tasks.ccre.baselines \
    --full_segments    data/hprc/full_segments.csv \
    --full_links       data/hprc/full_links.csv \
    --node_labels      data/hprc/ccre/run_001/node_labels.csv.gz \
    --test_chrs        chr8 chr19 chr22 \
    --val_chr          chr16 \
    --out_dir          results/hprc/ccre_baseline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from graph.io import read_segments_csv, read_links_csv
from graph.slicing import build_global_index
from graph.features import build_oid_metadata_from_segments
from graph.neg_sampling import oriented_ids_from_links
from utils.versioning import resolve_run_dir
from tasks.ccre.encoding import CCRE_CLASSES, CCRE_CLASS_TO_IDX


def _norm_chrom(c: str) -> str:
    """Accept both 'chr22' and 'GRCh38#0#chr22' in --test_chrs / --val_chr."""
    if c.startswith("chr"):
        return c
    if "#" in c:
        return c.split("#")[-1]
    return c


def build_per_node_structural_features(
    segids: np.ndarray,
    md: dict,
    deg_map: dict,
    drop_is_grch38: bool = True,
) -> np.ndarray:
    """Build (N, D) feature matrix per SEGMENT (not oriented). Uses forward
    orientation (oid = 2*segid) — for GRCh38 walks, forward vs reverse carry
    identical cCRE labels, so we pick one deterministically.

    Features (6 dims after dropping is_grch38):
        0: log1p(SO)              — genomic offset
        1: log1p(LN)              — segment length
        2: SR (raw)               — sample rank (constant for GRCh38 walks)
        3: log1p(degree)          — local degree in oriented space
        4: orientation bit        — 0 (we fix fwd)
        5: is_grch38 (optional, dropped for cCRE task to avoid shortcut)
    """
    oid_to_so = md["oid_to_so"]
    oid_to_ln = md["oid_to_ln"]
    oid_to_sr = md["oid_to_sr"]
    oid_to_is_grch38 = md["oid_to_is_grch38"]

    fwd_oids = (segids * 2).astype(np.int64)

    so = np.array([oid_to_so.get(int(o), 0) for o in fwd_oids], dtype=np.float64)
    ln = np.array([oid_to_ln.get(int(o), 1) for o in fwd_oids], dtype=np.float64)
    sr = np.array([oid_to_sr.get(int(o), 0) for o in fwd_oids], dtype=np.float32)
    deg = np.array(
        [deg_map.get(int(o), 0) + deg_map.get(int(o) + 1, 0) for o in fwd_oids],
        dtype=np.float64,
    )
    is_ref = np.array(
        [oid_to_is_grch38.get(int(o), 0) for o in fwd_oids], dtype=np.float32
    )

    cols = [
        np.log1p(np.abs(so)),
        np.log1p(ln),
        sr.astype(np.float64),
        np.log1p(deg),
        np.zeros_like(so),  # orientation fixed to fwd
    ]
    if not drop_is_grch38:
        cols.append(is_ref.astype(np.float64))
    X = np.stack(cols, axis=1).astype(np.float32)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_segments", required=True)
    ap.add_argument("--full_links", required=True)
    ap.add_argument(
        "--node_labels", required=True, help="Output of scripts.10_encode_ccre_to_nodes"
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--test_chrs",
        nargs="+",
        required=True,
        help="Chromosomes to hold out for test (e.g. chr8 chr19)",
    )
    ap.add_argument(
        "--val_chr",
        default=None,
        help="One chromosome for validation; must be disjoint from --test_chrs "
        "for publication runs. If unset, uses the first test chr for quick "
        "iteration.",
    )
    ap.add_argument(
        "--downsample_background",
        type=float,
        default=1.0,
        help="Keep this fraction of background nodes during "
        "training (≤1.0). Evaluation uses full set.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--keep_is_grch38",
        action="store_true",
        help="Keep is_grch38 feature (NOT recommended; becomes "
        "constant when training on GRCh38 walks only).",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = resolve_run_dir(Path(args.out_dir))
    print(f"[11] output dir: {out_dir}")

    test_chrs = set(_norm_chrom(c) for c in args.test_chrs)
    val_chr = _norm_chrom(args.val_chr) if args.val_chr else sorted(test_chrs)[0]
    if val_chr in test_chrs:
        print(
            "[11] WARNING: val_chr is also in test_chrs. This is acceptable for "
            "quick iteration but not for final paper runs."
        )
    print(f"[11] test_chrs: {sorted(test_chrs)}  val_chr: {val_chr}")

    # Load segments + links
    print(f"[11] Loading segments...")
    segments = read_segments_csv(args.full_segments)
    seg_index, seg_u = build_global_index(segments)
    md = build_oid_metadata_from_segments(segments, seg_index)

    print(f"[11] Loading links...")
    links = read_links_csv(args.full_links)
    u, v = oriented_ids_from_links(links, seg_index)

    # Global degree map (over ALL oriented edges)
    print(f"[11] Computing global oriented degree map...")
    deg_map: dict = {}
    for a in u.tolist():
        deg_map[int(a)] = deg_map.get(int(a), 0) + 1
    for b in v.tolist():
        deg_map[int(b)] = deg_map.get(int(b), 0) + 1

    # Load node labels
    print(f"[11] Loading node labels from {args.node_labels}")
    node_labels = pd.read_csv(args.node_labels, compression="infer")
    print(f"[11] Labeled segments: {len(node_labels):,}")
    print(f"[11] Label distribution:")
    print(node_labels["ccre_class"].value_counts().to_string())

    # Normalize chrom column
    node_labels["chrom"] = node_labels["chrom"].astype(str)

    # Build features
    print(f"[11] Building features ...")
    segids_all = node_labels["segid"].to_numpy(np.int64)
    X_all = build_per_node_structural_features(
        segids_all, md, deg_map, drop_is_grch38=not args.keep_is_grch38
    )
    y_all = node_labels["ccre_label"].to_numpy(np.int64)
    chrom_all = node_labels["chrom"].to_numpy()
    print(f"[11] Feature matrix: {X_all.shape}")

    # Split by chromosome
    is_test = np.isin(chrom_all, list(test_chrs))
    is_val = chrom_all == val_chr
    is_train = ~(is_test | is_val)
    print(
        f"[11] split sizes: train={is_train.sum():,}  "
        f"val={is_val.sum():,}  test={is_test.sum():,}"
    )

    X_train, y_train = X_all[is_train], y_all[is_train]
    X_val, y_val = X_all[is_val], y_all[is_val]
    X_test, y_test = X_all[is_test], y_all[is_test]
    chrom_test = chrom_all[is_test]

    # Optional downsampling of background class
    if args.downsample_background < 1.0:
        bg = y_train == CCRE_CLASS_TO_IDX["background"]
        n_bg = bg.sum()
        n_keep = int(n_bg * args.downsample_background)
        bg_idx = np.where(bg)[0]
        drop_idx = rng.choice(bg_idx, size=n_bg - n_keep, replace=False)
        keep_mask = np.ones(len(y_train), dtype=bool)
        keep_mask[drop_idx] = False
        X_train = X_train[keep_mask]
        y_train = y_train[keep_mask]
        print(
            f"[11] After background downsample {args.downsample_background:.2f}: "
            f"train={len(y_train):,}"
        )

    # Scale + fit
    print(f"[11] Fitting multinomial LR ...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=500,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    clf.fit(X_train_s, y_train)

    # Evaluate
    y_pred_test = clf.predict(X_test_s)
    y_pred_val = clf.predict(X_val_s)

    macro_f1_test = f1_score(y_test, y_pred_test, average="macro")
    macro_f1_val = f1_score(y_val, y_pred_val, average="macro")

    print(f"\n[11] === RESULTS ===")
    print(f"  val  macro-F1 (chr={val_chr}): {macro_f1_val:.4f}")
    print(f"  test macro-F1 ({sorted(test_chrs)}): {macro_f1_test:.4f}")

    # Per-class breakdown
    target_names = CCRE_CLASSES
    labels_order = list(range(len(CCRE_CLASSES)))
    print("\n[11] Classification report (TEST):")
    print(
        classification_report(
            y_test,
            y_pred_test,
            labels=labels_order,
            target_names=target_names,
            digits=3,
            zero_division=0,
        )
    )

    # Per-chromosome macro-F1 on test
    rows = []
    for c in sorted(set(chrom_test.tolist())):
        m = chrom_test == c
        if m.sum() == 0:
            continue
        f1 = f1_score(y_test[m], y_pred_test[m], average="macro", zero_division=0)
        rows.append({"chrom": c, "n_nodes": int(m.sum()), "macro_f1": float(f1)})
    per_chr = pd.DataFrame(rows).sort_values("chrom")
    print("\n[11] Per-chromosome test macro-F1:")
    print(per_chr.to_string(index=False))

    # Save artifacts
    per_chr_path = out_dir / "per_chrom_macrof1.csv"
    per_chr.to_csv(per_chr_path, index=False)
    print(f"[11] Saved {per_chr_path}")

    # Confusion matrix (test)
    cm = confusion_matrix(y_test, y_pred_test, labels=labels_order)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_path = out_dir / "confusion_test.csv"
    cm_df.to_csv(cm_path)
    print(f"[11] Saved {cm_path}")

    # Save predictions (small files if test is large)
    preds_path = out_dir / "preds_test.csv.gz"
    pd.DataFrame(
        {
            "segid": segids_all[is_test],
            "chrom": chrom_test,
            "y_true": y_test,
            "y_pred": y_pred_test,
        }
    ).to_csv(preds_path, index=False, compression="gzip")
    print(f"[11] Saved {preds_path}")

    summary = {
        "args": vars(args),
        "test_chrs": sorted(test_chrs),
        "val_chr": val_chr,
        "macro_f1_val": float(macro_f1_val),
        "macro_f1_test": float(macro_f1_test),
        "per_chrom_test_macro_f1": per_chr.to_dict(orient="records"),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "feature_names": [
            "log1p_SO",
            "log1p_LN",
            "SR",
            "log1p_deg",
            "orient_fwd",
        ]
        + (["is_grch38"] if args.keep_is_grch38 else []),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[11] Saved summary.json")


if __name__ == "__main__":
    main()

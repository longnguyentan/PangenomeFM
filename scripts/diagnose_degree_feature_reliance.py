#!/usr/bin/env python3
"""Does a trained v1 checkpoint rely on the masking degree shortcut?

The v1 encoder receives node degree measured on the *unmasked* slice while
messages flow over the *masked* graph, so a positive endpoint shows a
mismatch that a matched negative does not.  This script scores the same
held-out candidates twice with the frozen checkpoint:

* ``as_trained``: v1 inputs (unmasked degree + component features);
* ``visible``: degree recomputed on the masked message-passing graph and the
  component feature zeroed (``--node_structure_source visible`` semantics).

A large drop from ``as_trained`` to ``visible`` means reconstruction accuracy
was driven by the shortcut rather than by neighbourhood structure.  Nothing is
retrained and no labels are used except to score.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluation.external import _build_model_from_checkpoint, _namespace_from_checkpoint
from evaluation.splits import normalize_chrom
from graph.features import build_oid_metadata_from_segments
from graph.slicing import build_global_index
from training.pretrain import evaluate_shared, load_slice


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="Path or glob; first sorted match is used")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--full-segments", required=True)
    ap.add_argument("--test-chrs", nargs="+", required=True)
    ap.add_argument("--closure", choices=["strict", "1hop"], required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-slices", type=int)
    ap.add_argument(
        "--splits", nargs="+", default=["test"], choices=["train", "val", "test"],
        help="Candidate splits of held-out slices (default matches the trainer's held-out test).",
    )
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    matches = sorted(glob.glob(args.checkpoint))
    if not matches:
        raise SystemExit(f"No checkpoint matches {args.checkpoint!r}")
    checkpoint = matches[0]
    out = Path(args.out_dir)
    if (out / "summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    segments = pd.read_csv(args.full_segments, compression="infer", usecols=["id", "name", "LN", "SN", "SO", "SR"])
    seg_index, _ = build_global_index(segments)
    md = build_oid_metadata_from_segments(segments, seg_index)
    manifest = pd.read_csv(args.manifest)
    tests = {normalize_chrom(c) for c in args.test_chrs}
    manifest = manifest[(manifest["closure"].astype(str) == args.closure)
                        & manifest["target_sn"].map(normalize_chrom).isin(tests)]
    if args.max_slices:
        manifest = manifest.head(args.max_slices)

    rows = []
    for mode in ("as_trained", "visible"):
        ns = _namespace_from_checkpoint(ckpt, seed=int(ckpt.get("args", {}).get("seed", 42)))
        ns.device = str(device)
        ns.mask_query_edges = True
        ns.canonical_conflict_policy = "exclude"
        ns.node_structure_source = "unmasked" if mode == "as_trained" else "visible"
        model, predictor = _build_model_from_checkpoint(ckpt, ns, device)
        slices = [s for s in (load_slice(r, seg_index, md, segments, ns) for _, r in manifest.iterrows()) if s is not None]
        preds: list = []
        for split in args.splits:
            evaluate_shared(model, predictor, slices, split=split, args=ns, prediction_rows=preds)
        frame = pd.DataFrame(preds)
        y, p = frame["y_true"].to_numpy(), frame["p_edge"].to_numpy()
        rows.append({"mode": mode, "n_slices": len(slices), "n_candidates": len(y),
                     "prevalence": float(y.mean()), "auroc": float(roc_auc_score(y, p)),
                     "auprc": float(average_precision_score(y, p))})
        frame.to_csv(out / f"predictions_{mode}.csv.gz", index=False, compression="gzip")
    summary = {"checkpoint": checkpoint, "closure": args.closure, "test_chrs": sorted(tests),
               "results": rows,
               "auprc_drop_visible_minus_as_trained": rows[1]["auprc"] - rows[0]["auprc"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

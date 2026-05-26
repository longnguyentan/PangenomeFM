"""Score candidate missing graph edges for HGSVC imputation/enhancement.

This is a runnable pilot for the application Prof suggested.  It uses a frozen
HPRC-trained link-prediction checkpoint to score non-edges from a benchmark's
``edge_pred`` files.  High-scoring non-edges are candidate graph adjacencies to
inspect or validate against a newer pangenome construction.

If a newer graph build is available, pass its links and segments to mark which
candidates are recovered by the newer build and compute precision at K.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from graph.features import build_oid_metadata_from_segments
from graph.io import read_links_csv, read_segments_csv
from graph.neg_sampling import oriented_ids_from_links
from graph.slicing import build_global_index
from training.pretrain import _compute_adaptive_window_k, load_slice, tensorize_slice
from evaluation.external import _build_model_from_checkpoint, _namespace_from_checkpoint
from utils.versioning import resolve_run_dir


def _orient_char(oid: int) -> str:
    return "+" if int(oid) % 2 == 0 else "-"


def _comparison_edge_names(
    *,
    comparison_segments: Path,
    comparison_links: Path,
) -> set[tuple[str, str, str, str]]:
    segments = read_segments_csv(comparison_segments)
    links = read_links_csv(comparison_links)
    seg_index, _ = build_global_index(segments)
    u, v = oriented_ids_from_links(links, seg_index)
    names = list(seg_index.astype(str))
    out: set[tuple[str, str, str, str]] = set()
    for u_oid, v_oid in zip(u.tolist(), v.tolist()):
        out.add((names[int(u_oid) // 2], _orient_char(int(u_oid)), names[int(v_oid) // 2], _orient_char(int(v_oid))))
    return out


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    out = {
        "n": int(len(y)),
        "positive_fraction": float(np.mean(y)) if len(y) else float("nan"),
        "brier": float(brier_score_loss(y, p)) if len(y) else float("nan"),
    }
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    return out


@torch.no_grad()
def run_imputation_scoring(
    *,
    checkpoint: str | Path,
    manifest: str | Path,
    full_segments: str | Path,
    out_dir: str | Path,
    closure: str,
    split: str,
    candidate_label: int,
    top_k: int,
    comparison_segments: str | Path | None,
    comparison_links: str | Path | None,
    device_name: str,
    seed: int,
) -> dict[str, object]:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required.")
    device = torch.device(device_name)
    out_dir = resolve_run_dir(Path(out_dir))
    checkpoint = Path(checkpoint)
    ckpt = torch.load(checkpoint, map_location=device)
    args = _namespace_from_checkpoint(ckpt, seed=seed)

    segments = read_segments_csv(full_segments)
    seg_index, _ = build_global_index(segments)
    md = build_oid_metadata_from_segments(segments, seg_index)
    seg_names = list(seg_index.astype(str))

    manifest_df = pd.read_csv(manifest)
    if closure != "all":
        manifest_df = manifest_df[manifest_df["closure"].astype(str) == closure]
    if split != "all":
        if "split" in manifest_df.columns:
            manifest_df = manifest_df[manifest_df["split"].astype(str) == split]
        else:
            print("[impute] manifest has no split column; using all rows")
    if manifest_df.empty:
        raise RuntimeError("No manifest rows after closure/split filtering.")

    comparison_edges = None
    if comparison_segments and comparison_links:
        comparison_edges = _comparison_edge_names(
            comparison_segments=Path(comparison_segments),
            comparison_links=Path(comparison_links),
        )
        print(f"[impute] loaded {len(comparison_edges):,} comparison edges")

    model, predictor = _build_model_from_checkpoint(ckpt, args, device)
    rows: list[dict[str, object]] = []
    calibration_y: list[np.ndarray] = []
    calibration_p: list[np.ndarray] = []

    for i, row in manifest_df.iterrows():
        sd_raw = load_slice(row, seg_index, md, segments, args)
        if sd_raw is None:
            continue
        sd = tensorize_slice(sd_raw, device, args)
        if args.adaptive_window:
            eff_wk = _compute_adaptive_window_k(
                sd["branching_frac"],
                args.adaptive_window_base,
                args.adaptive_window_alpha,
            )
            for layer in model.linear_layers:
                layer.window_k = eff_wk

        h = model.encode_nodes(
            sd["X"],
            sd["so"],
            sd["src"],
            sd["dst"],
            sd["temps"],
            sd["edge_attr"],
            sd["orient"],
            sd["pop_ids"],
        )
        if predictor is not None:
            logits = predictor(h[sd["q_u"]], h[sd["q_v"]])
        else:
            logits = model.edge_predictor(torch.cat([h[sd["q_u"]], h[sd["q_v"]]], dim=-1)).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        labels = sd["labels"].cpu().numpy().astype(np.int64)
        calibration_y.append(labels)
        calibration_p.append(probs)

        q_u = sd_raw["nodes"][sd_raw["query_u"]]
        q_v = sd_raw["nodes"][sd_raw["query_v"]]
        keep = labels == candidate_label
        for u_oid, v_oid, label, score in zip(q_u[keep], q_v[keep], labels[keep], probs[keep]):
            u_name = seg_names[int(u_oid) // 2]
            v_name = seg_names[int(v_oid) // 2]
            cand_key = (u_name, _orient_char(int(u_oid)), v_name, _orient_char(int(v_oid)))
            out = {
                "slice": sd_raw["name"],
                "target_sn": sd_raw["target_sn"],
                "closure": sd_raw["closure"],
                "u_oid": int(u_oid),
                "v_oid": int(v_oid),
                "u_seg": u_name,
                "u_orient": cand_key[1],
                "v_seg": v_name,
                "v_orient": cand_key[3],
                "old_label": int(label),
                "p_edge": float(score),
            }
            if comparison_edges is not None:
                out["present_in_comparison"] = cand_key in comparison_edges
            rows.append(out)
        if (i + 1) % 25 == 0:
            print(f"[impute] processed {i + 1} slices; candidates={len(rows):,}")

    candidates = pd.DataFrame(rows).sort_values("p_edge", ascending=False)
    candidates.to_csv(out_dir / "candidate_edges.csv.gz", index=False, compression="gzip")
    if top_k > 0:
        candidates.head(top_k).to_csv(out_dir / f"candidate_edges_top{top_k}.csv", index=False)

    y = np.concatenate(calibration_y) if calibration_y else np.array([])
    p = np.concatenate(calibration_p) if calibration_p else np.array([])
    summary: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "manifest": str(manifest),
        "closure": closure,
        "split": split,
        "candidate_label": int(candidate_label),
        "n_candidates": int(len(candidates)),
        "top_k": int(top_k),
        "calibration_metrics_on_edge_pred": _metrics(y, p) if len(y) else {},
    }
    if comparison_edges is not None and "present_in_comparison" in candidates.columns:
        comp = candidates["present_in_comparison"].astype(bool).to_numpy()
        summary["comparison_n_present"] = int(comp.sum())
        summary["comparison_precision_all"] = float(comp.mean()) if len(comp) else float("nan")
        precision_at = {}
        for k in [10, 50, 100, 500, 1000, top_k]:
            if k and len(comp):
                precision_at[str(k)] = float(comp[: min(k, len(comp))].mean())
        summary["comparison_precision_at_k"] = precision_at
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Score candidate missing graph edges for imputation.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--full_segments", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--closure", choices=["strict", "1hop", "all"], default="1hop")
    ap.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    ap.add_argument(
        "--candidate_label",
        type=int,
        default=0,
        help="Which edge_pred label to score as imputation candidates; 0 means sampled non-edges.",
    )
    ap.add_argument("--top_k", type=int, default=1000)
    ap.add_argument("--comparison_segments", default=None)
    ap.add_argument("--comparison_links", default=None)
    ap.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run_imputation_scoring(
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        full_segments=args.full_segments,
        out_dir=args.out_dir,
        closure=args.closure,
        split=args.split,
        candidate_label=args.candidate_label,
        top_k=args.top_k,
        comparison_segments=args.comparison_segments,
        comparison_links=args.comparison_links,
        device_name=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

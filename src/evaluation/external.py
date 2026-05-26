"""Frozen external replication evaluation for link prediction.

This module loads a checkpoint produced by ``training.pretrain`` and evaluates
it, without tuning, on a benchmark manifest from an independent graph such as
HGSVC3. It reports AUROC plus class-balance-sensitive metrics requested in the
PSB plan: average precision and Brier score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from graph.features import build_oid_metadata_from_segments
from graph.io import read_segments_csv
from graph.slicing import build_global_index
from models.dual_stream_gat import DualStreamPangenomeGAT
from training.pretrain import (
    ExpressiveLinkPredictor,
    _compute_adaptive_window_k,
    load_slice,
    tensorize_slice,
)
from utils.versioning import resolve_run_dir


def _namespace_from_checkpoint(ckpt: Dict, seed: int) -> argparse.Namespace:
    train_args = dict(ckpt.get("args", {}))
    defaults = {
        "adaptive_window": False,
        "adaptive_window_base": 32,
        "adaptive_window_alpha": 4.0,
        "batch_size": 512,
        "dropout": 0.1,
        "dual_stream": True,
        "expressive_predictor": False,
        "hidden_dim": 48,
        "multiscale_rope": False,
        "n_heads": 4,
        "n_layers": 2,
        "n_rope_scales": 3,
        "no_fusion_gate": False,
        "no_rope": False,
        "orientation_rope": False,
        "pop_cond": False,
        "pop_embed_dim": 16,
        "seed": seed,
        "stream_mode": "full",
        "use_edge_features": False,
        "window_k": None,
    }
    defaults.update(train_args)
    defaults["seed"] = seed
    return argparse.Namespace(**defaults)


def _build_model_from_checkpoint(ckpt: Dict, args: argparse.Namespace, device):
    edge_feat_dim = int(ckpt.get("edge_feat_dim", 0))
    window_k = args.window_k if args.window_k is not None else (64 if args.adaptive_window else None)
    model = DualStreamPangenomeGAT(
        in_dim=int(ckpt["in_dim"]),
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        edge_mlp_dim=args.hidden_dim * 2,
        edge_feat_dim=edge_feat_dim,
        use_rope=not args.no_rope,
        use_fusion_gate=not args.no_fusion_gate,
        window_k=window_k,
        use_multiscale_rope=args.multiscale_rope,
        n_rope_scales=args.n_rope_scales,
        use_orientation=args.orientation_rope,
        use_cross_attn=False,
        pop_embed_dim=args.pop_embed_dim if args.pop_cond else 0,
        stream_mode=getattr(args, "stream_mode", ckpt.get("stream_mode", "full")),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    predictor = None
    if ckpt.get("predictor_state") is not None:
        predictor = ExpressiveLinkPredictor(
            hidden_dim=args.hidden_dim,
            mlp_dim=args.hidden_dim * 2,
            dropout=args.dropout,
        ).to(device)
        predictor.load_state_dict(ckpt["predictor_state"])

    model.eval()
    if predictor is not None:
        predictor.eval()
    return model, predictor


def _metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    out = {
        "n_edges": int(len(y)),
        "pos_frac": float(np.mean(y)) if len(y) else float("nan"),
        "brier": float(brier_score_loss(y, p)) if len(y) else float("nan"),
    }
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


@torch.no_grad()
def _score_slice(
    model: "DualStreamPangenomeGAT",
    predictor: Optional["ExpressiveLinkPredictor"],
    sd: Dict,
    split: str,
    args: argparse.Namespace,
) -> Tuple[Dict, pd.DataFrame]:
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

    if split == "all":
        idx = torch.arange(len(sd["labels"]), dtype=torch.long, device=sd["labels"].device)
    else:
        idx = sd[f"{split}_idx"]

    qu = sd["q_u"][idx]
    qv = sd["q_v"][idx]
    labels = sd["labels"][idx]
    if predictor is not None:
        logits = predictor(h[qu], h[qv])
    else:
        logits = model.edge_predictor(torch.cat([h[qu], h[qv]], dim=-1)).squeeze(-1)
    probs = torch.sigmoid(logits).cpu().numpy()
    y = labels.cpu().numpy()

    row = {
        "name": sd["name"],
        "target_sn": sd["target_sn"],
        "chrom": str(sd["target_sn"]).split("#")[-1].split("|")[-1],
        "closure": sd["closure"],
        "n_nodes": sd["n_nodes"],
        "branching_frac": float(sd["branching_frac"]),
        **_metrics(y, probs),
    }
    preds = pd.DataFrame(
        {
            "slice": sd["name"],
            "target_sn": sd["target_sn"],
            "closure": sd["closure"],
            "u_local": qu.cpu().numpy(),
            "v_local": qv.cpu().numpy(),
            "y_true": y,
            "p_edge": probs,
        }
    )
    return row, preds


def run_external_eval(
    *,
    checkpoint: str | Path,
    manifest: str | Path,
    full_segments: str | Path,
    out_dir: str | Path,
    closure: str | None,
    split: str,
    seed: int,
    device_name: str,
) -> Dict:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required.")

    device = torch.device(device_name)
    checkpoint = Path(checkpoint)
    manifest = Path(manifest)
    out_dir = resolve_run_dir(Path(out_dir))

    ckpt = torch.load(checkpoint, map_location=device)
    eval_args = _namespace_from_checkpoint(ckpt, seed=seed)
    ckpt_closure = str(ckpt.get("closure", "all"))
    requested_closure = closure or ckpt_closure

    print(f"[external] checkpoint: {checkpoint}")
    print(f"[external] checkpoint closure: {ckpt_closure}")
    print(f"[external] evaluation closure: {requested_closure}")
    print(f"[external] split: {split}")
    print(f"[external] output: {out_dir}")

    segments = read_segments_csv(full_segments)
    seg_index, _ = build_global_index(segments)
    md = build_oid_metadata_from_segments(segments, seg_index)

    manifest_df = pd.read_csv(manifest)
    if requested_closure != "all":
        manifest_df = manifest_df[manifest_df["closure"].astype(str) == requested_closure]
    if manifest_df.empty:
        raise RuntimeError(f"No manifest rows matched closure={requested_closure!r}.")

    raw_slices: List[Dict] = []
    for _, row in manifest_df.iterrows():
        sd = load_slice(row, seg_index, md, segments, eval_args)
        if sd is not None:
            raw_slices.append(sd)
    if not raw_slices:
        raise RuntimeError("No evaluable slices loaded from external benchmark.")

    model, predictor = _build_model_from_checkpoint(ckpt, eval_args, device)
    slices = [tensorize_slice(sd, device, eval_args) for sd in raw_slices]

    rows: List[Dict] = []
    pred_frames: List[pd.DataFrame] = []
    for sd in slices:
        row, preds = _score_slice(model, predictor, sd, split, eval_args)
        rows.append(row)
        pred_frames.append(preds)

    per_slice = pd.DataFrame(rows)
    pooled_preds = pd.concat(pred_frames, ignore_index=True)
    pooled_metrics = _metrics(
        pooled_preds["y_true"].to_numpy(np.float32),
        pooled_preds["p_edge"].to_numpy(np.float32),
    )
    by_target = (
        per_slice.groupby(["target_sn", "closure"])
        .agg(
            n_slices=("name", "count"),
            n_edges=("n_edges", "sum"),
            mean_auroc=("auroc", "mean"),
            mean_auprc=("auprc", "mean"),
            mean_brier=("brier", "mean"),
        )
        .reset_index()
    )

    per_slice.to_csv(out_dir / "per_slice_metrics.csv", index=False)
    by_target.to_csv(out_dir / "per_target_metrics.csv", index=False)
    pooled_preds.to_csv(out_dir / "pooled_predictions.csv.gz", index=False, compression="gzip")

    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_closure": ckpt_closure,
        "closure": requested_closure,
        "split": split,
        "seed": seed,
        "n_slices": int(len(per_slice)),
        "pooled_metrics": pooled_metrics,
        "per_target_metrics": by_target.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[external] pooled metrics:")
    print(json.dumps(pooled_metrics, indent=2))
    print("[external] per target:")
    print(by_target.to_string(index=False))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Frozen external checkpoint evaluation.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--full_segments", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--closure", default=None, choices=["strict", "1hop", "all"])
    ap.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = ap.parse_args()

    run_external_eval(
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        full_segments=args.full_segments,
        out_dir=args.out_dir,
        closure=args.closure,
        split=args.split,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()

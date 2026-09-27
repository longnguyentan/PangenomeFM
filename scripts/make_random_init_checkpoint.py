#!/usr/bin/env python3
"""Create a random-initialisation control from a trained PangenomeFM checkpoint.

The control keeps the exact architecture, node inputs, held-out chromosomes and
extraction behaviour of the source checkpoint, but replaces every encoder and
scorer weight with a fresh seeded initialisation.  Running the unchanged
frozen probes on it answers: does pretraining add information beyond a
random message-passing projection of the same inputs?  (Random GNN features
are a well-known strong baseline; a foundation-model claim needs T > R.)
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from evaluation.external import _build_model_from_checkpoint, _namespace_from_checkpoint


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ns = _namespace_from_checkpoint(ckpt, seed=args.seed)
    trained, _ = _build_model_from_checkpoint(ckpt, ns, torch.device("cpu"))
    torch.manual_seed(args.seed)
    fresh = type(trained)(**_constructor_kwargs(trained, ckpt, ns))
    out = dict(ckpt)
    out["model_state"] = fresh.state_dict()
    if ckpt.get("predictor_state") is not None:
        out["predictor_state"] = None  # scorer is irrelevant for frozen extraction
    out["random_init_control"] = {
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "seed": args.seed,
    }
    out["best_val_auc"] = float("nan")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"wrote {args.out}")
    return 0


def _constructor_kwargs(model, ckpt, ns) -> dict:
    window_k = ns.window_k if ns.window_k is not None else (64 if ns.adaptive_window else None)
    return dict(
        in_dim=int(ckpt["in_dim"]),
        hidden_dim=ns.hidden_dim,
        n_heads=ns.n_heads,
        n_layers=ns.n_layers,
        dropout=ns.dropout,
        edge_mlp_dim=ns.hidden_dim * 2,
        edge_feat_dim=int(ckpt.get("edge_feat_dim", 0)),
        use_rope=not ns.no_rope,
        use_fusion_gate=not ns.no_fusion_gate,
        window_k=window_k,
        use_multiscale_rope=ns.multiscale_rope,
        n_rope_scales=ns.n_rope_scales,
        use_orientation=ns.orientation_rope,
        use_cross_attn=False,
        pop_embed_dim=ns.pop_embed_dim if ns.pop_cond else 0,
        stream_mode=getattr(ns, "stream_mode", ckpt.get("stream_mode", "full")),
        graph_message_direction=getattr(ns, "graph_message_direction", "incoming"),
    )


if __name__ == "__main__":
    raise SystemExit(main())

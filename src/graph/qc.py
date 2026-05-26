from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .io import load_json, read_links_csv, read_segments_csv


@dataclass
class VerifyReport:
    base: str
    ok: bool
    issues: List[str]
    stats: Dict


def _edge_endpoints_exist(segments_sub: pd.DataFrame, links_sub: pd.DataFrame) -> bool:
    seg_names = set(segments_sub["name"].astype(str).tolist())
    return (
        links_sub["from_seg"].astype(str).isin(seg_names).all()
        and links_sub["to_seg"].astype(str).isin(seg_names).all()
    )


def _has_both_classes(edge_pred: pd.DataFrame) -> bool:
    if "label" not in edge_pred.columns:
        return False
    vc = edge_pred["label"].value_counts().to_dict()
    return (0 in vc) and (1 in vc) and vc[0] > 0 and vc[1] > 0


def verify_slice_triplet(
    seg_path: str | Path,
    link_path: str | Path,
    meta_path: str | Path,
    edge_pred_path: str | Path | None = None,
) -> VerifyReport:
    seg_path = Path(seg_path)
    link_path = Path(link_path)
    meta_path = Path(meta_path)
    base = seg_path.name.replace("_segments.csv.gz", "")

    issues: List[str] = []
    stats: Dict = {}
    ok = True

    seg = read_segments_csv(seg_path)
    links = read_links_csv(link_path)
    meta = load_json(meta_path)

    # 1) schema sanity
    for col in ["name", "SN", "SO", "LN", "SR"]:
        if col not in seg.columns:
            ok = False
            issues.append(f"Missing segments col: {col}")

    # 2) edge endpoints exist in segments_sub
    if not _edge_endpoints_exist(seg, links):
        ok = False
        issues.append(
            "Some link endpoints not present in segments_sub['name'] (broken induced subgraph)."
        )

    # 3) overlap should be all 0M in your current extraction
    if "overlap_all_0M" in meta and meta["overlap_all_0M"]:
        if not (links["overlap"].astype(str) == "0M").all():
            ok = False
            issues.append("meta says overlap_all_0M but links have non-0M overlap.")

    # 4) strict closure SN purity
    if meta.get("closure") == "strict":
        uniq_sn = seg["SN"].astype(str).nunique()
        if uniq_sn != 1:
            ok = False
            issues.append(f"strict slice should be SN-pure but has uniq_sn={uniq_sn}.")

    # 5) basic graph stats
    stats["n_segments"] = int(len(seg))
    stats["n_links"] = int(len(links))
    if len(links) > 0:
        deg = (
            pd.concat([links["from_seg"], links["to_seg"]])
            .astype("string")
            .value_counts()
        )
        stats["max_degree"] = int(deg.max())
        stats["branching_frac_deg_gt_2"] = float((deg > 2).mean())
        stats["has_cycle_signal_proxy"] = bool((deg > 10).any())  # proxy only
    else:
        stats["max_degree"] = 0
        stats["branching_frac_deg_gt_2"] = 0.0
        stats["has_cycle_signal_proxy"] = False

    # 6) meta match (not strict equality, but check key fields exist)
    for k in [
        "type",
        "target_sn",
        "start",
        "end",
        "window_bp",
        "closure",
        "n_segments",
        "n_links",
    ]:
        if k not in meta:
            ok = False
            issues.append(f"meta missing key: {k}")

    # 7) edge_pred dataset validity
    if edge_pred_path is not None:
        edge_pred_path = Path(edge_pred_path)
        edge_pred = pd.read_csv(edge_pred_path, compression="infer")
        if not _has_both_classes(edge_pred):
            ok = False
            issues.append(
                "edge_pred has missing/empty class (need both label=0 and label=1)."
            )
        stats["edge_pred_rows"] = int(len(edge_pred))

    return VerifyReport(base=base, ok=ok, issues=issues, stats=stats)


def verify_manifest(manifest_path: str | Path) -> pd.DataFrame:
    """
    Verify all entries in a manifest.csv produced by scripts/00_make_benchmark_v1.py.
    """
    manifest_path = Path(manifest_path)
    m = pd.read_csv(manifest_path)

    rows = []
    for _, r in m.iterrows():
        rep = verify_slice_triplet(
            r["segments_path"],
            r["links_path"],
            r["meta_path"],
            r.get("edge_pred_path", None),
        )
        rows.append(
            {
                "name": r["name"],
                "target_sn": r["target_sn"],
                "closure": r["closure"],
                "ok": rep.ok,
                "issues": " | ".join(rep.issues),
                **rep.stats,
            }
        )
    return pd.DataFrame(rows)

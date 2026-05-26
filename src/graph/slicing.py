from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class SliceResult:
    segments_sub: pd.DataFrame
    links_sub: pd.DataFrame
    meta: Dict


def build_global_index(segments: pd.DataFrame) -> Tuple[pd.Index, pd.DataFrame]:
    """
    Canonicalize segment ordering so that:
      - seg_index = unique segment names (canonical ID space)
      - seg_u aligned to seg_index (one row per segment name)
    This MUST be used everywhere (window selection, link mapping, slicing).
    """
    seg_index = pd.Index(segments["name"].astype("string").unique())
    seg_u = (
        segments.drop_duplicates("name").set_index("name").loc[seg_index].reset_index()
    )
    # seg_u columns now: name, id, seq, LN, SN, SO, SR (order depends on input)
    return seg_index, seg_u


def map_links_to_segids(
    links: pd.DataFrame, seg_index: pd.Index
) -> Tuple[np.ndarray, np.ndarray]:
    from_id = seg_index.get_indexer(links["from_seg"].astype("string")).astype(np.int64)
    to_id = seg_index.get_indexer(links["to_seg"].astype("string")).astype(np.int64)
    if (from_id < 0).any() or (to_id < 0).any():
        bad_from = links.loc[from_id < 0, "from_seg"].head().tolist()
        bad_to = links.loc[to_id < 0, "to_seg"].head().tolist()
        raise ValueError(
            f"Link endpoints missing in seg_index. from={bad_from} to={bad_to}"
        )
    return from_id, to_id


def segids_in_window_by_sn(
    SN_by_segid: np.ndarray,
    SO_by_segid: np.ndarray,
    LN_by_segid: np.ndarray,
    target_sn: str,
    start: int,
    end: int,
) -> np.ndarray:
    sn_mask = SN_by_segid == target_sn
    overlap = sn_mask & (SO_by_segid < end) & ((SO_by_segid + LN_by_segid) > start)
    return np.where(overlap)[0].astype(np.int64)


def build_sn_interval_table(
    seg_u: pd.DataFrame, target_sn: str
) -> Optional[pd.DataFrame]:
    sn_df = seg_u[seg_u["SN"].astype("string") == target_sn][["SO", "LN"]].copy()
    if len(sn_df) == 0:
        return None
    sn_df["END"] = sn_df["SO"].astype(np.int64) + sn_df["LN"].astype(np.int64)
    sn_df = sn_df.sort_values("SO").reset_index(drop=True)
    return sn_df


def choose_window_gap_aware(
    sn_df: pd.DataFrame,
    window_bp: int,
    rng: np.random.Generator,
    max_tries: int = 300,
    prefer_dense: bool = True,
):
    so = sn_df["SO"].to_numpy(np.int64)
    endp = sn_df["END"].to_numpy(np.int64)

    best = None
    best_count = -1

    for _ in range(max_tries):
        j = int(rng.integers(0, len(sn_df)))
        anchor = int(so[j])
        start = anchor - int(rng.integers(0, window_bp))
        end = start + window_bp
        mask = (so < end) & (endp > start)
        cnt = int(mask.sum())

        if cnt > 0 and not prefer_dense:
            return int(start), int(end), cnt

        if cnt > best_count:
            best_count = cnt
            best = (int(start), int(end), cnt)

    if best is None or best_count <= 0:
        return None
    return best  # (start,end,cnt)


def induced_subgraph(
    segments: pd.DataFrame,
    links: pd.DataFrame,
    seg_index: pd.Index,
    segids_core: np.ndarray,
    add_one_hop: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Build induced subgraph on seg_index ID space.
    strict: only core nodes.
    1hop: add nodes touching core by any edge.
    """
    from_id, to_id = map_links_to_segids(links, seg_index)

    core_mask = np.zeros(len(seg_index), dtype=bool)
    core_mask[segids_core] = True

    if add_one_hop:
        touch = core_mask[from_id] | core_mask[to_id]
        segids_touch = np.unique(np.concatenate([from_id[touch], to_id[touch]])).astype(
            np.int64
        )
        keep_mask = np.zeros(len(seg_index), dtype=bool)
        keep_mask[segids_touch] = True
    else:
        keep_mask = core_mask

    edge_mask = keep_mask[from_id] & keep_mask[to_id]
    links_sub = links.loc[edge_mask].copy().reset_index(drop=True)

    segids_final = np.where(keep_mask)[0].astype(np.int64)
    seg_names_sub = seg_index[segids_final]
    segments_sub = (
        segments[segments["name"].astype("string").isin(seg_names_sub)]
        .copy()
        .reset_index(drop=True)
    )

    # closure check
    if len(links_sub) > 0:
        assert links_sub["from_seg"].isin(segments_sub["name"]).all()
        assert links_sub["to_seg"].isin(segments_sub["name"]).all()

    return segments_sub, links_sub, segids_final


def qc_slice(
    segments_sub: pd.DataFrame, links_sub: pd.DataFrame, topk: int = 10
) -> Dict:
    out: Dict = {
        "n_segments": int(len(segments_sub)),
        "n_links": int(len(links_sub)),
        "SR_counts_top": segments_sub["SR"].value_counts().head(topk).to_dict(),
        "SN_counts_top": segments_sub["SN"]
        .astype("string")
        .value_counts()
        .head(topk)
        .to_dict(),
    }

    if len(links_sub) > 0:
        deg = (
            pd.concat([links_sub["from_seg"], links_sub["to_seg"]])
            .astype("string")
            .value_counts()
        )
        out["max_degree"] = int(deg.max())
        out["branching_frac_deg_gt_2"] = float((deg > 2).mean())
        out["from_orient_counts"] = (
            links_sub["from_orient"].astype(str).value_counts().to_dict()
        )
        out["to_orient_counts"] = (
            links_sub["to_orient"].astype(str).value_counts().to_dict()
        )
    else:
        out["max_degree"] = 0
        out["branching_frac_deg_gt_2"] = 0.0
        out["from_orient_counts"] = {}
        out["to_orient_counts"] = {}

    # stringify keys for JSON friendliness
    out["SR_counts_top"] = {str(k): int(v) for k, v in out["SR_counts_top"].items()}
    out["SN_counts_top"] = {str(k): int(v) for k, v in out["SN_counts_top"].items()}
    out["from_orient_counts"] = {
        str(k): int(v) for k, v in out["from_orient_counts"].items()
    }
    out["to_orient_counts"] = {
        str(k): int(v) for k, v in out["to_orient_counts"].items()
    }
    return out

"""
src/tasks/ccre/encoding.py

ENCODE cCRE BED parsing and GRCh38-walk node mapping.

Inputs
------
ENCODE SCREEN v4 candidate cis-Regulatory Elements BED file (6 cols):
    chrom  start  end  cCRE_accession  RDHS_id  cCRE_class

cCRE classes (9 including "background"):
    dELS        - distal enhancer-like signature
    pELS        - proximal enhancer-like signature
    PLS         - promoter-like signature
    CA          - chromatin accessible
    CA-CTCF     - chromatin accessible with CTCF binding
    CA-H3K4me3  - chromatin accessible with H3K4me3
    CA-TF       - chromatin accessible with TF binding
    TF          - TF binding only
    background  - no cCRE overlap (assigned to nodes we label as "no cCRE")

Mapping rules
-------------
1. Only GRCh38-walk nodes can carry a cCRE label (labels exist on GRCh38 coords).
2. A cCRE can overlap multiple graph nodes (nodes are variable length).
3. A node can overlap multiple cCREs; we assign the DOMINANT class
   (argmax overlap_bp). Ties broken by class priority (PLS > pELS > dELS >
   CA-CTCF > CA-H3K4me3 > CA-TF > TF > CA).
4. A node with zero cCRE overlap gets label "background".
5. Both orientations (oid = 2*segid and oid = 2*segid+1) inherit the same
   label — cCRE identity is strand-neutral for the 150-350bp scale of cCREs.

This module provides:
    * parse_encode_bed()                : 6-col BED → DataFrame
    * normalize_chrom()                 : "chr22" ↔ "GRCh38#0#chr22"
    * map_ccre_to_ref_nodes()           : sort-merge overlap, O(N+M)
    * CCRE_CLASSES, CCRE_CLASS_TO_IDX   : canonical ordering (for softmax)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Canonical class ordering - MUST NOT CHANGE (used as softmax label space)
# ---------------------------------------------------------------------------

CCRE_CLASSES: List[str] = [
    "background",  # 0 - no overlap
    "PLS",  # 1
    "pELS",  # 2
    "dELS",  # 3
    "CA-CTCF",  # 4
    "CA-H3K4me3",  # 5
    "CA-TF",  # 6
    "TF",  # 7
    "CA",  # 8
]
CCRE_CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CCRE_CLASSES)}
N_CCRE_CLASSES: int = len(CCRE_CLASSES)

# Priority for tie-breaking when a node overlaps cCREs of multiple classes with
# identical overlap_bp. Functionally meaningful classes win over generic CA.
_CLASS_PRIORITY: Dict[str, int] = {
    "PLS": 100,
    "pELS": 90,
    "dELS": 80,
    "CA-CTCF": 70,
    "CA-H3K4me3": 60,
    "CA-TF": 50,
    "TF": 40,
    "CA": 30,
    "background": 0,
}


# ---------------------------------------------------------------------------
# BED parsing
# ---------------------------------------------------------------------------


def parse_encode_bed(path: str | Path) -> pd.DataFrame:
    """Parse a 6-column ENCODE cCRE BED. Returns DataFrame with columns:
    chrom, start, end, cCRE_acc, RDHS_id, cCRE_class.
    """
    p = Path(path)
    df = pd.read_csv(
        p,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "cCRE_acc", "RDHS_id", "cCRE_class"],
        dtype={
            "chrom": str,
            "start": np.int64,
            "end": np.int64,
            "cCRE_acc": str,
            "RDHS_id": str,
            "cCRE_class": str,
        },
    )
    unknown = set(df["cCRE_class"].unique()) - set(CCRE_CLASSES)
    if unknown:
        raise ValueError(
            f"Unknown cCRE class(es) in BED: {unknown}. "
            f"Expected one of {CCRE_CLASSES}"
        )
    return df


# ---------------------------------------------------------------------------
# Chromosome name normalization
# ---------------------------------------------------------------------------


def sn_to_chrom(sn: str) -> Optional[str]:
    """Extract 'chr22' from an SN tag like 'GRCh38#0#chr22' or 'chr22'.
    Returns None if SN does not appear to be a GRCh38-like reference walk."""
    if sn.startswith("chr"):
        return sn
    if "#" in sn:
        parts = sn.split("#")
        last = parts[-1]
        if last.startswith("chr"):
            return last
    return None


def chrom_to_sn_candidates(
    chrom: str, sn_universe: List[str], ref_only: bool = True
) -> List[str]:
    """Find SN tags in `sn_universe` whose chromosome matches `chrom`.

    By default (`ref_only=True`), only returns SNs starting with 'GRCh38' —
    this is what we want for cCRE mapping, since cCRE labels only exist on
    GRCh38 coordinates. Pass `ref_only=False` to include alt haplotypes
    (e.g. HG002, CHM13) — useful for bubble-based label propagation later.
    """
    hits = [s for s in sn_universe if sn_to_chrom(s) == chrom]
    if ref_only:
        hits = [s for s in hits if s.startswith("GRCh38")]
    return hits


# ---------------------------------------------------------------------------
# Sort-merge overlap between cCREs and ref-walk segments
# ---------------------------------------------------------------------------


def _overlap_bp(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _accumulate_segment_class_overlaps(
    ccre_df: pd.DataFrame,  # single chrom, sorted by start
    seg_df: pd.DataFrame,  # single SN, sorted by SO; cols: segid, SO, LN
) -> Dict[int, Dict[str, int]]:
    """
    For every segment, return overlap_bp per class.
    Sweep: two pointers. For each segment, advance ccre pointer past all
    cCREs whose end <= seg_start; scan active cCREs whose start < seg_end.
    """
    ccre_starts = ccre_df["start"].to_numpy()
    ccre_ends = ccre_df["end"].to_numpy()
    ccre_classes = ccre_df["cCRE_class"].to_numpy()

    seg_so = seg_df["SO"].to_numpy()
    seg_ln = seg_df["LN"].to_numpy()
    seg_ids = seg_df["segid"].to_numpy()

    result: Dict[int, Dict[str, int]] = {}
    ccre_ptr = 0
    n_ccre = len(ccre_df)

    for i in range(len(seg_df)):
        s_start = int(seg_so[i])
        s_end = s_start + int(seg_ln[i])
        if s_end <= s_start:
            continue

        # Advance ccre_ptr past cCREs that end before this segment starts
        while ccre_ptr < n_ccre and int(ccre_ends[ccre_ptr]) <= s_start:
            ccre_ptr += 1

        # Scan forward through cCREs that start before this segment ends.
        # We do NOT advance ccre_ptr (cCREs can overlap multiple segments).
        j = ccre_ptr
        per_class: Dict[str, int] = {}
        while j < n_ccre and int(ccre_starts[j]) < s_end:
            ov = _overlap_bp(int(ccre_starts[j]), int(ccre_ends[j]), s_start, s_end)
            if ov > 0:
                cls = str(ccre_classes[j])
                per_class[cls] = per_class.get(cls, 0) + ov
            j += 1

        if per_class:
            result[int(seg_ids[i])] = per_class

    return result


def assign_dominant_class(per_class: Dict[str, int]) -> Tuple[str, int]:
    """Return (dominant_class, total_overlap_bp) from a per-class overlap dict.
    Ties broken by _CLASS_PRIORITY."""
    if not per_class:
        return "background", 0
    best_cls = max(
        per_class.items(),
        key=lambda kv: (kv[1], _CLASS_PRIORITY.get(kv[0], 0)),
    )[0]
    return best_cls, sum(per_class.values())


def map_ccre_to_ref_nodes(
    ccre_df: pd.DataFrame,
    seg_u: pd.DataFrame,  # from build_global_index; cols: name, seq, LN, SN, SO, SR
    seg_index: pd.Index,  # from build_global_index
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Map ENCODE cCRE intervals onto GRCh38-walk graph segments.

    Returns DataFrame with one row per SEGMENT (not per oid; duplicate each
    row to both orientations downstream):
        segid, oid_fwd, oid_rev, SN, SO, LN, ccre_class, overlap_bp,
        dominant_class_overlap_bp, n_ccres_touching

    Non-GRCh38 segments are NOT included here (they get label "background"
    or UNK from the caller).
    """
    # Index segments with their row position so we can recover segid later
    seg_u = seg_u.reset_index(drop=True).copy()
    seg_u["segid"] = np.arange(len(seg_u), dtype=np.int64)

    # Only GRCh38 walks
    is_ref = seg_u["SN"].astype(str).str.startswith("GRCh38")
    ref_seg = seg_u[is_ref].reset_index(drop=True)
    sn_universe = sorted(ref_seg["SN"].astype(str).unique())
    if verbose:
        print(
            f"[encode_ccre] GRCh38 walks found: {len(sn_universe)} "
            f"({len(ref_seg):,} segments)"
        )

    # Group cCREs and segments by chrom/SN for sort-merge
    all_rows: List[Dict] = []

    for chrom in sorted(ccre_df["chrom"].unique()):
        sn_list = chrom_to_sn_candidates(chrom, sn_universe)
        if not sn_list:
            if verbose:
                print(f"  [skip] {chrom}: no matching SN walk in graph")
            continue

        ccre_sub = (
            ccre_df[ccre_df["chrom"] == chrom]
            .sort_values("start")
            .reset_index(drop=True)
        )
        if len(ccre_sub) == 0:
            continue

        for sn in sn_list:
            seg_sub = (
                ref_seg[ref_seg["SN"].astype(str) == sn][["segid", "SO", "LN"]]
                .sort_values("SO")
                .reset_index(drop=True)
            )
            if len(seg_sub) == 0:
                continue

            overlaps = _accumulate_segment_class_overlaps(ccre_sub, seg_sub)

            # Emit one row per segment (even for background, so we keep coverage)
            ln_lookup = dict(zip(seg_sub["segid"].tolist(), seg_sub["LN"].tolist()))
            so_lookup = dict(zip(seg_sub["segid"].tolist(), seg_sub["SO"].tolist()))
            for segid in seg_sub["segid"].tolist():
                per_class = overlaps.get(int(segid), {})
                dom_cls, total_ov = assign_dominant_class(per_class)
                dom_ov = per_class.get(dom_cls, 0) if dom_cls != "background" else 0
                all_rows.append(
                    {
                        "segid": int(segid),
                        "oid_fwd": int(segid) * 2 + 0,
                        "oid_rev": int(segid) * 2 + 1,
                        "SN": sn,
                        "chrom": chrom,
                        "SO": int(so_lookup[int(segid)]),
                        "LN": int(ln_lookup[int(segid)]),
                        "ccre_class": dom_cls,
                        "overlap_bp": int(total_ov),
                        "dominant_class_overlap_bp": int(dom_ov),
                        "n_ccres_touching": int(len(per_class)),
                    }
                )

            if verbose:
                labeled = sum(
                    1
                    for r in all_rows[-len(seg_sub) :]
                    if r["ccre_class"] != "background"
                )
                print(
                    f"  [{chrom}] {sn}: {len(seg_sub):,} segs, "
                    f"{labeled:,} labeled (non-bg)"
                )

    out = pd.DataFrame(all_rows)
    if len(out) == 0:
        raise RuntimeError(
            "map_ccre_to_ref_nodes produced zero rows. "
            "Likely chromosome name mismatch between BED and SN tags."
        )
    return out


def summarize_label_table(node_labels: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab of chrom × ccre_class counts + totals."""
    ct = pd.crosstab(node_labels["chrom"], node_labels["ccre_class"])
    # Enforce canonical column order; missing columns filled with 0
    for cls in CCRE_CLASSES:
        if cls not in ct.columns:
            ct[cls] = 0
    ct = ct[CCRE_CLASSES]
    ct["TOTAL"] = ct.sum(axis=1)
    return ct

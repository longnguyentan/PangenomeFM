from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def oid_to_segid_bit(oid: int) -> Tuple[int, int]:
    return int(oid // 2), int(oid % 2)


def build_oid_metadata_from_segments(
    segments: pd.DataFrame, seg_index: pd.Index
) -> Dict[str, Dict[int, object]]:
    """
    Build dictionaries for oid -> SN, SO, LN, SR, is_reference.
    Note: assumes seg_index aligns to full segments row order.

    The historical key is still named `oid_to_is_grch38` for compatibility with
    older training code, but the value now means "reference walk" rather than
    literally GRCh38. This keeps HPRC behavior unchanged while letting CHM13
    HGSVC3 reference nodes use the same feature convention.
    """
    sn = segments["SN"].astype(str).to_numpy()
    so = segments["SO"].to_numpy(dtype=np.int64)
    ln = segments["LN"].to_numpy(dtype=np.int64)
    sr = segments["SR"].to_numpy()

    oid_to_sn: Dict[int, str] = {}
    oid_to_so: Dict[int, int] = {}
    oid_to_ln: Dict[int, int] = {}
    oid_to_sr: Dict[int, int] = {}
    oid_to_is_grch38: Dict[int, int] = {}

    for segid in range(len(seg_index)):
        base_sn = str(sn[segid])
        base_so = int(so[segid])
        base_ln = int(ln[segid])
        base_sr = int(sr[segid])
        is_ref = 1 if base_sr == 0 or base_sn.startswith(("GRCh38", "CHM13", "id=CHM13")) else 0
        for bit in (0, 1):
            oid = segid * 2 + bit
            oid_to_sn[oid] = base_sn
            oid_to_so[oid] = base_so
            oid_to_ln[oid] = base_ln
            oid_to_sr[oid] = base_sr
            oid_to_is_grch38[oid] = is_ref

    return {
        "oid_to_sn": oid_to_sn,
        "oid_to_so": oid_to_so,
        "oid_to_ln": oid_to_ln,
        "oid_to_sr": oid_to_sr,
        "oid_to_is_grch38": oid_to_is_grch38,
    }


def edge_features(
    u: np.ndarray,
    v: np.ndarray,
    oid_to_so: Dict[int, int],
    oid_to_ln: Dict[int, int],
    oid_to_sr: Dict[int, int],
    oid_to_is_grch38: Dict[int, int],
    deg_map: Dict[int, int],
) -> pd.DataFrame:
    """
    Features used in your ablations:
      - delta_so (absolute coord difference)
      - deg (deg_u + deg_v)
      - same_sn (computed later by comparing oid_to_sn)
      - sr (sr_u, sr_v)
      - len (ln_u, ln_v)
      - orient (bit_u, bit_v)
      - is_grch38 (is_grch38_u, is_grch38_v)
    """
    uu = u.astype(int)
    vv = v.astype(int)

    delta_so = np.array(
        [abs(oid_to_so[int(a)] - oid_to_so[int(b)]) for a, b in zip(uu, vv)],
        dtype=np.int64,
    )
    deg = np.array(
        [deg_map.get(int(a), 0) + deg_map.get(int(b), 0) for a, b in zip(uu, vv)],
        dtype=np.int32,
    )

    sr_u = np.array([oid_to_sr[int(a)] for a in uu], dtype=np.int32)
    sr_v = np.array([oid_to_sr[int(b)] for b in vv], dtype=np.int32)

    ln_u = np.array([oid_to_ln[int(a)] for a in uu], dtype=np.int32)
    ln_v = np.array([oid_to_ln[int(b)] for b in vv], dtype=np.int32)

    bit_u = (uu % 2).astype(np.int8)
    bit_v = (vv % 2).astype(np.int8)

    is_grch38_u = np.array([oid_to_is_grch38[int(a)] for a in uu], dtype=np.int8)
    is_grch38_v = np.array([oid_to_is_grch38[int(b)] for b in vv], dtype=np.int8)

    return pd.DataFrame(
        {
            "delta_so": delta_so,
            "deg": deg,
            "sr_u": sr_u,
            "sr_v": sr_v,
            "ln_u": ln_u,
            "ln_v": ln_v,
            "orient_u": bit_u,
            "orient_v": bit_v,
            "is_grch38_u": is_grch38_u,
            "is_grch38_v": is_grch38_v,
        }
    )

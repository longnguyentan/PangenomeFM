from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def read_segments_csv(path: str | Path) -> pd.DataFrame:
    """
    Expected columns (parsed from GFA):
      id, name, seq, LN, SN, SO, SR
    Notes:
      - seq can be large; keep as string
      - SN is sample/contig identifier
      - SO is offset coordinate (int)
      - LN is length (int)
      - SR is rank / source-related tag (int-like)
    """
    path = Path(path)
    df = pd.read_csv(path, compression="infer")
    # soft-validate required cols
    req = {"id", "name", "seq", "LN", "SN", "SO", "SR"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Segments missing columns: {missing} in {path}")
    return df


def read_links_csv(path: str | Path) -> pd.DataFrame:
    """
    Expected columns (parsed from GFA):
      from_seg, from_orient, to_seg, to_orient, overlap, SR, L1, L2
    """
    path = Path(path)
    df = pd.read_csv(path, compression="infer")
    req = {"from_seg", "from_orient", "to_seg", "to_orient", "overlap"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Links missing columns: {missing} in {path}")
    return df


def save_gz_csv(df: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, compression="gzip")


def save_json(obj: Dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def load_json(path: str | Path) -> Dict:
    path = Path(path)
    return json.loads(path.read_text())


def infer_out_prefix(out_dir: str | Path, name: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / name


def list_slice_triplets(out_dir: str | Path):
    """
    Utility: find *_segments.csv.gz, *_links.csv.gz, *_meta.json triplets.
    """
    out_dir = Path(out_dir)
    seg_files = sorted(out_dir.glob("*_segments.csv.gz"))
    for segf in seg_files:
        base = segf.name.replace("_segments.csv.gz", "")
        linkf = out_dir / f"{base}_links.csv.gz"
        metaf = out_dir / f"{base}_meta.json"
        if linkf.exists() and metaf.exists():
            yield base, segf, linkf, metaf

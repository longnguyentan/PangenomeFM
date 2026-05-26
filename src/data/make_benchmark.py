"""
Build benchmark windows and link-prediction manifests from cleaned graph tables.

Primary usage:
    python -m graphgenomefm make-benchmark --data-dir data/hprc

Direct module usage:
    python -m data.make_benchmark \\
        --data_dir data/hprc \\
        --out_dir data/hprc/benchmark \\
        --segments full_segments.csv \\
        --links full_links.csv
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

from graph.io import save_gz_csv, save_json
from graph.slicing import (
    build_global_index,
    build_sn_interval_table,
    choose_window_gap_aware,
    segids_in_window_by_sn,
    induced_subgraph,
    qc_slice,
)
from graph.neg_sampling import (
    oriented_ids_from_links,
    build_pos_set,
    slice_oriented_node_set,
    compute_oriented_degrees,
    neg_random,
    neg_hard_coord_degree,
    neg_distance_matched,
)
from graph.features import build_oid_metadata_from_segments

SEGMENT_REQUIRED = {"id", "name", "seq", "LN", "SN", "SO", "SR"}
LINK_REQUIRED = {"from_seg", "from_orient", "to_seg", "to_orient", "overlap"}
CANONICAL_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Find files
# ─────────────────────────────────────────────────────────────────────────────


def _cols_ok(path: Path, required: set) -> Tuple[bool, set]:
    try:
        df = pd.read_csv(path, nrows=2, compression="infer")
        missing = required - set(df.columns)
        return len(missing) == 0, missing
    except Exception:
        return False, required


def find_data_files(
    data_dir: Path,
    seg_name: Optional[str],
    link_name: Optional[str],
) -> Tuple[Path, Path]:

    print("\n" + "=" * 60)
    print("STEP 1 — Locating data files")
    print("=" * 60)

    csvs = sorted(data_dir.glob("*.csv")) + sorted(data_dir.glob("*.csv.gz"))
    print(f"  Files in {data_dir}:")
    for f in csvs:
        print(f"    {f.name}")

    # ── Segments ─────────────────────────────────────────────────────────────
    seg_path: Optional[Path] = None

    if seg_name:
        p = data_dir / seg_name
        if not p.exists():
            print(f"\n  ✗ --segments '{seg_name}' not found in {data_dir}")
            sys.exit(1)
        ok, miss = _cols_ok(p, SEGMENT_REQUIRED)
        if not ok:
            print(f"  ✗ {seg_name} missing columns: {miss}")
            sys.exit(1)
        seg_path = p
        print(f"\n  ✓ Segments (explicit): {seg_name}")
    else:
        # FIX 1: full_segments.csv added as first priority
        seg_priority = [
            "full_segments.csv",
            "segments_parsed_half.csv",
            "segments_parsed_SR_larger_than_0.csv",
            "segments_parsed.csv",
            "segments.csv",
        ]
        for name in seg_priority:
            p = data_dir / name
            if p.exists():
                ok, _ = _cols_ok(p, SEGMENT_REQUIRED)
                if ok:
                    seg_path = p
                    print(f"\n  ✓ Segments (auto): {name}")
                    break
        if seg_path is None:
            for f in csvs:
                if "seg" in f.name.lower():
                    ok, _ = _cols_ok(f, SEGMENT_REQUIRED)
                    if ok:
                        seg_path = f
                        print(f"\n  ✓ Segments (fallback): {f.name}")
                        break

    if seg_path is None:
        print(f"\n  ✗ No valid segments file found. Required cols: {SEGMENT_REQUIRED}")
        sys.exit(1)

    # ── Links ─────────────────────────────────────────────────────────────────
    link_path: Optional[Path] = None

    if link_name:
        p = data_dir / link_name
        if not p.exists():
            print(f"\n  ✗ --links '{link_name}' not found in {data_dir}")
            sys.exit(1)
        ok, miss = _cols_ok(p, LINK_REQUIRED)
        if not ok:
            print(f"  ✗ {link_name} missing columns: {miss}")
            sys.exit(1)
        link_path = p
        print(f"  ✓ Links    (explicit): {link_name}")
    else:
        # FIX 1: full_links.csv added as first priority
        link_priority = [
            "full_links.csv",
            "links_parsed_half.csv",
            "links_parsed.csv",
            "links.csv",
        ]
        for name in link_priority:
            p = data_dir / name
            if p.exists():
                ok, _ = _cols_ok(p, LINK_REQUIRED)
                if ok:
                    link_path = p
                    print(f"  ✓ Links    (auto): {name}")
                    break
        if link_path is None:
            for f in csvs:
                if "link" in f.name.lower():
                    ok, _ = _cols_ok(f, LINK_REQUIRED)
                    if ok:
                        link_path = f
                        print(f"  ✓ Links    (fallback): {f.name}")
                        break

    if link_path is None:
        print(f"\n  ✗ No valid links file found. Required cols: {LINK_REQUIRED}")
        sys.exit(1)

    return seg_path, link_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Inspect
# ─────────────────────────────────────────────────────────────────────────────


def inspect_data(
    seg_path: Path,
    link_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:

    print("\n" + "=" * 60)
    print("STEP 2 — Inspecting data")
    print("=" * 60)

    segs = pd.read_csv(seg_path, compression="infer")
    links = pd.read_csv(link_path, compression="infer")

    print(f"\n  Segments: {len(segs):,} rows × {segs.shape[1]} cols")
    print(f"  Links:    {len(links):,} rows × {links.shape[1]} cols")
    print(f"  Segment columns: {list(segs.columns)}")
    print(f"  Links columns:   {list(links.columns)}")

    sn_counts = segs["SN"].astype(str).value_counts()
    n_unique_sn = segs["SN"].nunique()
    print(f"\n  Unique SN values: {n_unique_sn:,}")
    print(f"  Top 10 SN values:")
    for sn, cnt in sn_counts.head(10).items():
        print(f"    {sn:55s}  {cnt:>8,} segments")
    if n_unique_sn > 10:
        print(f"    ... ({n_unique_sn - 10:,} more not shown)")

    if "SR" in segs.columns:
        sr_counts = segs["SR"].value_counts().sort_index()
        n_sr = len(sr_counts)
        # FIX 2: cap SR output to avoid wall of text with large files
        print(f"\n  SR distribution ({n_sr} unique values):")
        shown = list(sr_counts.items())[:20]
        for sr, cnt in shown:
            print(f"    SR={sr}: {cnt:,} segments")
        if n_sr > 20:
            print(f"    ... ({n_sr - 20} more SR values not shown)")

    # Validate link endpoints exist in segments
    seg_names = set(segs["name"].astype(str))
    link_from = set(links["from_seg"].astype(str).unique())
    link_to = set(links["to_seg"].astype(str).unique())
    all_link_nodes = link_from | link_to
    missing_nodes = all_link_nodes - seg_names
    pct_missing = 100 * len(missing_nodes) / max(len(all_link_nodes), 1)

    print(f"\n  Link endpoint check:")
    print(f"    Unique nodes in links:    {len(all_link_nodes):,}")
    print(f"    Of those in segments:     {len(all_link_nodes - missing_nodes):,}")
    print(f"    Missing from segments:    {len(missing_nodes):,}  ({pct_missing:.1f}%)")
    if missing_nodes:
        sample = sorted(missing_nodes)[:5]
        print(f"    Sample missing: {sample}")
        if pct_missing > 50:
            print("\n  ✗ More than 50% of link endpoints are missing from segments.")
            print("    Your links and segments files likely do not match.")
            sys.exit(1)

    # Choose target SNs: all canonical reference chromosomes when possible.
    sn_set = {str(s) for s in sn_counts.index}
    grch = [f"GRCh38#0#{chrom}" for chrom in CANONICAL_CHROMS]
    chm13_hash = [f"CHM13#0#{chrom}" for chrom in CANONICAL_CHROMS]
    chm13_pipe = [f"id=CHM13|{chrom}" for chrom in CANONICAL_CHROMS]
    if any(s in sn_set for s in grch):
        targets = [s for s in grch if s in sn_set]
        target_note = "canonical GRCh38 chromosomes"
    elif any(s in sn_set for s in chm13_hash):
        targets = [s for s in chm13_hash if s in sn_set]
        target_note = "canonical CHM13 chromosomes"
    elif any(s in sn_set for s in chm13_pipe):
        targets = [s for s in chm13_pipe if s in sn_set]
        target_note = "canonical CHM13 chromosomes"
    else:
        targets = sn_counts.index[:3].astype(str).tolist()
        target_note = "top 3 SN values"

    print(f"\n  ✓ Auto-detected target SNs ({target_note}): {targets}")
    print(f"    (override with --targets if needed)")

    return segs, links, targets


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Build manifest
# ─────────────────────────────────────────────────────────────────────────────


def build_manifest(
    segs: pd.DataFrame,
    links: pd.DataFrame,
    targets: List[str],
    out_dir: Path,
    seed: int,
    window_bp: int,
    n_windows: int,
    closures: List[str],
    negative_sampler: str = "random",
    negative_same_sn: bool = True,
    negative_coord_band: int = 5_000,
    negative_tol_bp: int = 1_000,
    negative_tol_frac: float = 0.10,
    negative_degree_matched: bool = False,
) -> pd.DataFrame:

    print("\n" + "=" * 60)
    print("STEP 3 — Building benchmark slices + manifest")
    print("=" * 60)
    print(
        f"  window_bp={window_bp:,}  n_windows={n_windows}  "
        f"closures={closures}  targets={len(targets)}"
    )
    print(
        f"  Expected slices: {len(targets)} × {n_windows} × {len(closures)} "
        f"= {len(targets)*n_windows*len(closures)}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    seg_index, seg_u = build_global_index(segs)
    md = build_oid_metadata_from_segments(segs, seg_index)
    SN_arr = seg_u["SN"].astype("string").to_numpy()
    SO_arr = seg_u["SO"].to_numpy(np.int64)
    LN_arr = seg_u["LN"].to_numpy(np.int64)

    # Pre-filter links: only keep edges whose both endpoints are in seg_index
    seg_name_set = set(seg_index.tolist())
    link_mask = links["from_seg"].astype(str).isin(seg_name_set) & links[
        "to_seg"
    ].astype(str).isin(seg_name_set)
    links_filtered = links[link_mask].reset_index(drop=True)
    dropped = len(links) - len(links_filtered)
    if dropped:
        print(f"\n  ⚠ Dropped {dropped:,} links whose endpoints are not in segments.")
    print(f"  Using {len(links_filtered):,} links for slicing.\n")

    manifest_rows: List[Dict] = []

    for target_sn in targets:
        print(f"  Target SN: {target_sn}")
        sn_df = build_sn_interval_table(seg_u, target_sn)
        if sn_df is None or len(sn_df) == 0:
            print(f"    ✗ No segments found — skipping.")
            continue

        for closure in closures:
            made = 0
            attempts = 0
            while made < n_windows and attempts < n_windows * 15:
                attempts += 1
                picked = choose_window_gap_aware(sn_df, window_bp, rng)
                if picked is None:
                    continue
                start, end, _ = picked

                segids_core = segids_in_window_by_sn(
                    SN_arr, SO_arr, LN_arr, target_sn, start, end
                )
                if len(segids_core) == 0:
                    continue

                segs_sub, links_sub, _ = induced_subgraph(
                    segments=segs,
                    links=links_filtered,
                    seg_index=seg_index,
                    segids_core=segids_core,
                    add_one_hop=(closure == "1hop"),
                )
                if len(links_sub) == 0:
                    continue

                qc = qc_slice(segs_sub, links_sub)
                safe = (
                    str(target_sn).replace("#", "_").replace("/", "_").replace(" ", "_")
                )
                name = f"slice_{safe}_{start}_{end}_{closure}"[:120]

                seg_out = out_dir / f"{name}_segments.csv.gz"
                link_out = out_dir / f"{name}_links.csv.gz"
                meta_out = out_dir / f"{name}_meta.json"
                edge_out = out_dir / f"{name}_edge_pred.csv.gz"

                save_gz_csv(segs_sub, seg_out)
                save_gz_csv(links_sub, link_out)
                save_json(
                    {
                        "type": "window",
                        "target_sn": str(target_sn),
                        "start": int(start),
                        "end": int(end),
                        "window_bp": window_bp,
                        "closure": closure,
                        "core_overlap_segments": int(len(segids_core)),
                        "overlap_all_0M": True,
                        "negative_sampler": negative_sampler,
                        "negative_same_sn": negative_same_sn,
                        "negative_coord_band": negative_coord_band,
                        "negative_tol_bp": negative_tol_bp,
                        "negative_tol_frac": negative_tol_frac,
                        "negative_degree_matched": negative_degree_matched,
                        **qc,
                    },
                    meta_out,
                )

                # Edge prediction dataset
                u, v = oriented_ids_from_links(links_sub, seg_index)
                pos = np.stack([u, v], axis=1)
                nodes = slice_oriented_node_set(u, v)
                pos_set = build_pos_set(u, v)
                rng_neg = np.random.default_rng(seed + made)
                if negative_sampler == "random":
                    neg = neg_random(
                        nodes,
                        pos_set,
                        n_neg=len(pos),
                        rng=rng_neg,
                    )
                elif negative_sampler == "hard_coord_degree":
                    deg_map = compute_oriented_degrees(u, v, nodes)
                    neg = neg_hard_coord_degree(
                        nodes=nodes,
                        pos_pairs=pos,
                        pos_set=pos_set,
                        oid_to_sn=md["oid_to_sn"],
                        oid_to_so=md["oid_to_so"],
                        oid_to_deg=deg_map,
                        n_neg=len(pos),
                        rng=rng_neg,
                        same_sn=negative_same_sn,
                        coord_band=negative_coord_band,
                        degree_matched=negative_degree_matched,
                    )
                elif negative_sampler == "distance_matched":
                    deg_map = compute_oriented_degrees(u, v, nodes)
                    neg = neg_distance_matched(
                        nodes=nodes,
                        pos_pairs=pos,
                        pos_set=pos_set,
                        oid_to_sn=md["oid_to_sn"],
                        oid_to_so=md["oid_to_so"],
                        oid_to_deg=deg_map,
                        n_neg=len(pos),
                        rng=rng_neg,
                        same_sn=negative_same_sn,
                        tol_bp=negative_tol_bp,
                        tol_frac=negative_tol_frac,
                        degree_matched=negative_degree_matched,
                    )
                else:
                    raise ValueError(f"Unknown negative_sampler={negative_sampler!r}")

                if len(neg) < len(pos):
                    print(
                        f"    ✗ {negative_sampler} produced only {len(neg)}/{len(pos)} "
                        "negatives — retrying another window."
                    )
                    continue

                neg = neg[: len(pos)]
                df_pos = pd.DataFrame(
                    {"u_oid": pos[:, 0], "v_oid": pos[:, 1], "label": 1}
                )
                df_neg = pd.DataFrame(
                    {
                        "u_oid": [a for a, b in neg],
                        "v_oid": [b for a, b in neg],
                        "label": 0,
                    }
                )
                (
                    pd.concat([df_pos, df_neg])
                    .sample(frac=1, random_state=seed)
                    .reset_index(drop=True)
                    .to_csv(edge_out, index=False, compression="gzip")
                )

                manifest_rows.append(
                    {
                        "slice_id": name,
                        "name": name,
                        "target_sn": target_sn,
                        "closure": closure,
                        "segments_path": str(seg_out),
                        "links_path": str(link_out),
                        "meta_path": str(meta_out),
                        "edge_pred_path": str(edge_out),
                        "n_segments": qc["n_segments"],
                        "n_links": qc["n_links"],
                        "negative_sampler": negative_sampler,
                        "negative_same_sn": negative_same_sn,
                        "negative_coord_band": negative_coord_band,
                        "negative_tol_bp": negative_tol_bp,
                        "negative_tol_frac": negative_tol_frac,
                        "negative_degree_matched": negative_degree_matched,
                    }
                )
                made += 1
                print(
                    f"    [{closure}] {made}/{n_windows}: "
                    f"{qc['n_segments']} segs, {qc['n_links']} links  "
                    f"({start}–{end}, negatives={negative_sampler})"
                )

    if not manifest_rows:
        print("\n  ✗ No slices built. Try --window_bp 20000 or a smaller value.")
        sys.exit(1)

    manifest = pd.DataFrame(manifest_rows)
    mpath = out_dir / "manifest.csv"
    manifest.to_csv(mpath, index=False)
    print(f"\n  ✓ Manifest saved: {mpath}  ({len(manifest)} slices total)")
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Network analysis
# ─────────────────────────────────────────────────────────────────────────────


def run_network_analysis(
    manifest: pd.DataFrame,
    segs: pd.DataFrame,
    seg_index,
    out_dir: Path,
    do_viz: bool,
) -> pd.DataFrame:

    print("\n" + "=" * 60)
    print("STEP 4 — Network analysis")
    print("=" * 60)

    import json
    from graph.features import build_oid_metadata_from_segments
    from analysis.network import analyze, build_adjacency, stats_to_dict
    from models.attention_window import (
        compute_branching_distances,
        compute_attention_temperature,
        summarize_attention_windows,
    )

    md = build_oid_metadata_from_segments(segs, seg_index)
    net_out = out_dir / "network_analysis"
    net_out.mkdir(parents=True, exist_ok=True)
    all_stats = []

    for _, row in manifest.iterrows():
        name = row["name"]
        links_sub = pd.read_csv(row["links_path"], compression="infer")
        if len(links_sub) == 0:
            continue

        u, v = oriented_ids_from_links(links_sub, seg_index)
        nodes = slice_oriented_node_set(u, v)

        stats = analyze(u, v, nodes, sample_paths=(len(nodes) < 5_000))
        adj = build_adjacency(u, v, nodes)
        oid_to_radius = compute_branching_distances(adj, stats.branching_nodes)
        window_summary = summarize_attention_windows(
            oid_to_radius, stats.branching_nodes
        )

        sd = {**stats_to_dict(stats), **window_summary, "name": name}
        with open(net_out / f"{name[:80]}_net.json", "w") as f:
            json.dump(sd, f, indent=2, default=str)

        print(
            f"  {name[:55]:55s}  "
            f"N={stats.n_nodes:5d}  branch={stats.branching_frac:.3f}  "
            f"γ={str(round(stats.degree_exponent,2)) if stats.degree_exponent else 'N/A':>5}  "
            f"attn_r={window_summary['mean_attention_radius']:.2f}"
        )

        if do_viz:
            try:
                from models.gat import build_node_features
                from analysis.latent import visualize_all

                deg_map = {int(n): 0 for n in nodes}
                for a in u:
                    deg_map[int(a)] += 1
                for b in v:
                    deg_map[int(b)] += 1
                X = build_node_features(
                    nodes=nodes,
                    oid_to_so=md["oid_to_so"],
                    oid_to_ln=md["oid_to_ln"],
                    oid_to_sr=md["oid_to_sr"],
                    oid_to_is_grch38=md["oid_to_is_grch38"],
                    oid_to_degree=deg_map,
                    oid_to_component_id=stats.oid_to_component_id,
                )
                nl = nodes.astype(int).tolist()
                visualize_all(
                    node_features=X,
                    nodes=nodes,
                    degrees=np.array([stats.oid_to_degree.get(n, 0) for n in nl]),
                    is_grch38=np.array(
                        [md["oid_to_is_grch38"].get(n, 0) for n in nl], dtype=float
                    ),
                    clustering_coeffs=np.array(
                        [stats.oid_to_clustering.get(n, 0.0) for n in nl]
                    ),
                    component_ids=np.array(
                        [stats.oid_to_component_id.get(n, 0) for n in nl]
                    ),
                    oid_to_radius=oid_to_radius,
                    hub_oids=stats.hub_oids,
                    out_dir=net_out / "viz" / name[:60],
                    label=name[:40],
                    use_umap=True,
                )
            except Exception as e:
                print(f"    [viz skipped: {e}]")

        all_stats.append(sd)

    summary = pd.DataFrame(all_stats)
    summary.to_csv(net_out / "network_summary.csv", index=False)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Print commands
# ─────────────────────────────────────────────────────────────────────────────


def print_next_steps(seg_path: Path, link_path: Path, manifest_path: Path, out_dir: Path):
    # Derive result dirs from out_dir.
    base = out_dir.parent / "results" / out_dir.name
    print("\n" + "=" * 60)
    print("STEP 5 — Commands to run next")
    print("=" * 60)
    print(
        f"""
Each training command auto-creates a versioned run_NNN/ subdir inside --out_dir.

── Shared pretraining ──────────────────────────────────────────────
python -m training.pretrain \\
    --manifest {manifest_path} \\
    --full_segments {seg_path} \\
    --out_dir {base}/pretrain \\
    --hidden_dim 48 --n_heads 4 --n_layers 2 \\
    --epochs 100 --patience 20 \\
    --dual_stream --adaptive_window --multiscale_rope --orientation_rope \\
    --focal_loss --drop_edge --warmup_epochs 5
"""
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Bootstrap the full pangenome-GAT pipeline from raw CSV files."
    )
    ap.add_argument(
        "--data_dir",
        default="data",
        help="Directory with your CSV files (default: data/)",
    )
    ap.add_argument(
        "--out_dir",
        default="data/benchmark",
        help="Output directory for slices + manifest (default: data/benchmark/)",
    )
    ap.add_argument(
        "--segments",
        default=None,
        help="Segments filename inside data_dir, e.g. full_segments.csv",
    )
    ap.add_argument(
        "--links",
        default=None,
        help="Links filename inside data_dir, e.g. full_links.csv",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window_bp", type=int, default=50_000)
    ap.add_argument("--n_windows", type=int, default=5)
    ap.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Force specific target SN values (default: auto-detect top 3)",
    )
    ap.add_argument(
        "--closures", nargs="+", default=["strict", "1hop"], choices=["strict", "1hop"]
    )
    ap.add_argument(
        "--negative_sampler",
        default="random",
        choices=["random", "hard_coord_degree", "distance_matched"],
        help=(
            "Negative edge sampler for edge_pred files. Use distance_matched "
            "for the paper-safety audit/rerun."
        ),
    )
    ap.add_argument(
        "--negative_same_sn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For hard samplers, constrain negative endpoints to the same sequence/chromosome.",
    )
    ap.add_argument(
        "--allow_cross_sn_negatives",
        action="store_false",
        dest="negative_same_sn",
        help="For hard samplers, allow negative endpoints to span different sequence names.",
    )
    ap.add_argument(
        "--negative_coord_band",
        type=int,
        default=5_000,
        help="Coordinate-delta tolerance for hard_coord_degree negatives.",
    )
    ap.add_argument(
        "--negative_tol_bp",
        type=int,
        default=1_000,
        help="Minimum bp tolerance for distance_matched negatives.",
    )
    ap.add_argument(
        "--negative_tol_frac",
        type=float,
        default=0.10,
        help="Fractional tolerance for distance_matched negatives.",
    )
    ap.add_argument(
        "--negative_degree_matched",
        action="store_true",
        help="Also match endpoint degree pairs for hard negative samplers.",
    )
    ap.add_argument("--no_network_analysis", action="store_true")
    ap.add_argument("--no_viz", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    seg_path, link_path = find_data_files(data_dir, args.segments, args.links)
    segs, links, auto_targets = inspect_data(seg_path, link_path)
    targets = args.targets if args.targets else auto_targets

    manifest = build_manifest(
        segs=segs,
        links=links,
        targets=targets,
        out_dir=out_dir,
        seed=args.seed,
        window_bp=args.window_bp,
        n_windows=args.n_windows,
        closures=args.closures,
        negative_sampler=args.negative_sampler,
        negative_same_sn=args.negative_same_sn,
        negative_coord_band=args.negative_coord_band,
        negative_tol_bp=args.negative_tol_bp,
        negative_tol_frac=args.negative_tol_frac,
        negative_degree_matched=args.negative_degree_matched,
    )

    if not args.no_network_analysis:
        seg_index, _ = build_global_index(segs)
        run_network_analysis(
            manifest=manifest,
            segs=segs,
            seg_index=seg_index,
            out_dir=out_dir,
            do_viz=not args.no_viz,
        )

    print_next_steps(seg_path, link_path, out_dir / "manifest.csv", out_dir)
    print("\n✓ Bootstrap complete.\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify degree baselines on the graph visible in each evaluation context.

For one deterministic seed, this audit covers the union of all five held-out
chromosome folds.  It reloads each closure-specific manifest row, reconstructs
the exact strict or one-hop adjacency used by the baseline evaluator, reapplies
positive query-edge masking, and compares recomputed degree-sum and
preferential-attachment scores with the archived raw scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.link_heuristics import _build_undirected_adjacency, _score_candidates
from graph.neg_sampling import oriented_ids_from_links
from graph.slicing import build_global_index
from scripts.server.evaluate_rotating_link_baselines import resolve_path


BASELINES = {
    "topology_degree_sum": "degree_sum",
    "topology_preferential_attachment": "preferential_attachment",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_archived_scores(baseline_root: Path, seed: int, closure: str) -> pd.DataFrame:
    paths = sorted(
        baseline_root.glob(
            f"hprc_r2/fold_*/seed_{seed}/{closure}/heldout_predictions.csv.gz"
        )
    )
    if len(paths) != 5:
        raise ValueError(
            f"expected five held-out baseline files for seed={seed}, closure={closure}; found {len(paths)}"
        )
    columns = [
        "slice",
        "closure",
        "candidate_row_index",
        "u_oid",
        "v_oid",
        "y_true",
        "baseline",
        "raw_score",
    ]
    frames: list[pd.DataFrame] = []
    for path in paths:
        kept: list[pd.DataFrame] = []
        for chunk in pd.read_csv(
            path,
            compression="infer",
            usecols=columns,
            chunksize=500_000,
        ):
            selected = chunk["baseline"].isin(BASELINES)
            if selected.any():
                kept.append(chunk.loc[selected].copy())
        if not kept:
            raise ValueError(f"no degree baseline rows in {path}")
        frames.append(pd.concat(kept, ignore_index=True))
    archived = pd.concat(frames, ignore_index=True)
    if not archived["closure"].astype(str).eq(closure).all():
        raise ValueError("archived closure metadata mismatch")
    keys = ["slice", "candidate_row_index", "u_oid", "v_oid", "y_true"]
    counts = archived.groupby(keys)["baseline"].nunique()
    if not counts.eq(len(BASELINES)).all():
        raise ValueError("degree baselines do not cover identical archived candidates")
    return archived


def audit_closure(
    *,
    manifest: pd.DataFrame,
    manifest_dir: Path,
    archived: pd.DataFrame,
    seg_index: dict[str, int],
    closure: str,
    max_slices: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_closure = manifest.loc[manifest["closure"].astype(str).eq(closure)].copy()
    if manifest_closure["name"].duplicated().any():
        raise ValueError(f"manifest names are not unique for {closure}")
    row_by_name = manifest_closure.set_index("name", drop=False)
    slice_names = sorted(archived["slice"].astype(str).unique())
    if max_slices > 0:
        slice_names = slice_names[:max_slices]
    checks: list[pd.DataFrame] = []
    exposure_rows: list[dict[str, object]] = []
    for index, slice_name in enumerate(slice_names, start=1):
        if slice_name not in row_by_name.index:
            raise KeyError(f"archived slice not present in {closure} manifest: {slice_name}")
        row = row_by_name.loc[slice_name]
        print(
            f"[visible-graph-audit] {closure} {index}/{len(slice_names)} {slice_name}",
            flush=True,
        )
        links_path = resolve_path(row["links_path"], manifest_dir)
        candidates_path = resolve_path(row["edge_pred_path"], manifest_dir)
        links = pd.read_csv(links_path, compression="infer")
        structural_u, structural_v = oriented_ids_from_links(links, seg_index)
        adjacency = _build_undirected_adjacency(zip(structural_u, structural_v))
        candidates = pd.read_csv(candidates_path, compression="infer")

        archived_slice = archived.loc[archived["slice"].astype(str).eq(slice_name)].copy()
        degree = archived_slice.loc[
            archived_slice["baseline"].eq("topology_degree_sum")
        ].sort_values("candidate_row_index")
        candidate_indices = degree["candidate_row_index"].to_numpy(np.int64)
        recomputed = _score_candidates(
            adjacency,
            candidates,
            candidate_indices,
            mask_query_edges=True,
            shortest_path_cutoff=8,
        )
        visible_nodes = len(adjacency)
        visible_edges = sum(len(neighbors) for neighbors in adjacency.values()) // 2
        exposure_rows.append(
            {
                "slice": slice_name,
                "closure": closure,
                "links_path": str(links_path.resolve()),
                "candidate_path": str(candidates_path.resolve()),
                "visible_oriented_nodes": int(visible_nodes),
                "visible_undirected_edges": int(visible_edges),
                "audited_candidates": int(len(candidate_indices)),
            }
        )
        for archived_name, score_name in BASELINES.items():
            stored = archived_slice.loc[
                archived_slice["baseline"].eq(archived_name)
            ].sort_values("candidate_row_index")
            expected_indices = stored["candidate_row_index"].to_numpy(np.int64)
            if not np.array_equal(expected_indices, candidate_indices):
                raise ValueError(f"candidate mismatch between degree baselines in {slice_name}")
            recalculated = recomputed[score_name].to_numpy(float)
            comparison = stored[
                ["slice", "candidate_row_index", "u_oid", "v_oid", "y_true"]
            ].copy()
            comparison["closure"] = closure
            comparison["baseline"] = archived_name
            comparison["stored_raw_score"] = stored["raw_score"].to_numpy(float)
            comparison["recomputed_raw_score"] = recalculated
            comparison["absolute_difference"] = np.abs(
                comparison["stored_raw_score"] - comparison["recomputed_raw_score"]
            )
            checks.append(comparison)
    return pd.concat(checks, ignore_index=True), pd.DataFrame(exposure_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-segments", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument(
        "--max-slices",
        type=int,
        default=0,
        help="Deterministic debugging limit; zero audits every held-out slice.",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    required = {"name", "closure", "links_path", "edge_pred_path"}
    if missing := sorted(required - set(manifest.columns)):
        raise ValueError(f"manifest is missing columns: {missing}")
    segments = pd.read_csv(
        args.full_segments,
        compression="infer",
        usecols=["name", "seq", "SN", "SO"],
    )
    seg_index, _ = build_global_index(segments)

    all_checks: list[pd.DataFrame] = []
    all_exposure: list[pd.DataFrame] = []
    for closure in ("strict", "1hop"):
        archived = load_archived_scores(args.baseline_root, args.seed, closure)
        checks, exposure = audit_closure(
            manifest=manifest,
            manifest_dir=args.manifest.parent,
            archived=archived,
            seg_index=seg_index,
            closure=closure,
            max_slices=args.max_slices,
        )
        all_checks.append(checks)
        all_exposure.append(exposure)
    checks = pd.concat(all_checks, ignore_index=True)
    exposure = pd.concat(all_exposure, ignore_index=True)
    maximum_difference = float(checks["absolute_difference"].max())
    failures = int((checks["absolute_difference"] > args.tolerance).sum())
    exposure_summary = (
        exposure.groupby("closure", sort=True)
        .agg(
            slices=("slice", "nunique"),
            median_visible_oriented_nodes=("visible_oriented_nodes", "median"),
            median_visible_undirected_edges=("visible_undirected_edges", "median"),
            audited_candidates=("audited_candidates", "sum"),
        )
        .reset_index()
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checks.to_csv(args.out_dir / "visible_graph_score_checks.csv.gz", index=False, compression="gzip")
    exposure.to_csv(args.out_dir / "visible_graph_slice_exposure.csv", index=False)
    exposure_summary.to_csv(args.out_dir / "visible_graph_exposure_summary.csv", index=False)
    audit = {
        "schema_version": 1,
        "status": "pass" if failures == 0 else "fail",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "baseline_root": str(args.baseline_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "full_segments": str(args.full_segments.resolve()),
        "seed": args.seed,
        "contexts": ["strict", "1hop"],
        "audited_slices": int(exposure["slice"].nunique()),
        "audited_score_rows": int(len(checks)),
        "baselines": sorted(BASELINES),
        "adjacency_source": "closure-specific links_path from the corresponding manifest row",
        "query_masking": "positive direct query edge removed independently before scoring each candidate",
        "tolerance": args.tolerance,
        "maximum_absolute_difference": maximum_difference,
        "failed_score_rows": failures,
        "debug_slice_limit": args.max_slices,
        "outputs": {
            "score_checks": "visible_graph_score_checks.csv.gz",
            "slice_exposure": "visible_graph_slice_exposure.csv",
            "exposure_summary": "visible_graph_exposure_summary.csv",
        },
    }
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

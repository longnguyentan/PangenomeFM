#!/usr/bin/env python3
"""Extract performance-independent graph features and freeze complexity thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from analysis.complexity import (
    FEATURE_DEFINITIONS,
    apply_complexity_definition,
    compute_slice_complexity,
    fit_complexity_definition,
)
from evaluation.splits import normalize_chrom


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str, manifest_path: Path) -> Path:
    candidate = Path(value)
    if candidate.exists():
        return candidate
    relative_to_manifest = manifest_path.parent / candidate
    if relative_to_manifest.exists():
        return relative_to_manifest
    raise FileNotFoundError(f"could not resolve {value!r} from {manifest_path}")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def extract(
    manifest_path: Path,
    *,
    dataset: str,
    contexts: set[str] | None = None,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    required = {
        "target_sn", "start", "end", "closure", "segments_path",
        "links_path", "edge_pred_path",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    if contexts is not None:
        manifest = manifest.loc[
            manifest["closure"].astype(str).isin(contexts)
        ].copy()
        if manifest.empty:
            raise ValueError(f"no manifest rows match contexts {sorted(contexts)}")
    rows: list[dict[str, object]] = []
    for row in manifest.itertuples(index=False):
        segments_path = resolve_path(str(row.segments_path), manifest_path)
        links_path = resolve_path(str(row.links_path), manifest_path)
        candidates_path = resolve_path(str(row.edge_pred_path), manifest_path)
        segments = pd.read_csv(segments_path, compression="infer")
        links = pd.read_csv(links_path, compression="infer")
        candidates = pd.read_csv(candidates_path, compression="infer")
        start, end = int(row.start), int(row.end)
        features = compute_slice_complexity(
            segments, links, candidates, window_bp=end - start
        )
        rows.append(
            {
                "slice_id": str(
                    getattr(row, "slice_id", None) or getattr(row, "name")
                ),
                "dataset": str(dataset),
                "target_sn": str(row.target_sn),
                "chromosome": normalize_chrom(str(row.target_sn)),
                "start": start,
                "end": end,
                "window_bp": end - start,
                "context": str(row.closure),
                "context_regime": str(
                    getattr(
                        row,
                        "context_regime",
                        "core-node-induced-subgraph"
                        if str(row.closure) == "strict"
                        else "endpoint-expanded-induced-subgraph",
                    )
                ),
                **features,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "data/hprc/benchmark_matched_nonoverlap_v2_review_20260804/manifest.csv",
    )
    parser.add_argument(
        "--dataset",
        default="hprc_local_matched_benchmark",
        help="Dataset/regime label written to every output row.",
    )
    parser.add_argument(
        "--contexts",
        nargs="+",
        choices=["strict", "1hop"],
        help=(
            "Optional contexts to extract. Use strict alone to freeze locus "
            "complexity without reading legacy expanded candidate files."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/complexity_definition_v1.yaml",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results/complexity/graph_window_complexity_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    protected = out_dir / "complexity_thresholds.json"
    if protected.exists() and not args.overwrite:
        raise FileExistsError(
            f"frozen thresholds already exist at {protected}; use --overwrite explicitly"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    definition = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    features = extract(
        args.manifest,
        dataset=args.dataset,
        contexts=set(args.contexts) if args.contexts else None,
    )
    reference_context = str(definition["fit_population"]["reference_context"])
    reference = features.loc[features["context"].eq(reference_context)].copy()
    if reference.empty:
        raise ValueError(f"no rows found for reference context {reference_context!r}")

    reference_scored, frozen = fit_complexity_definition(reference, definition)
    observed_scored = apply_complexity_definition(features, frozen)
    observed_scored = observed_scored.rename(
        columns={
            "complexity_score": "observed_context_complexity_score",
            "complexity_category": "observed_context_complexity_category",
        }
    )

    locus_columns = ["dataset", "target_sn", "start", "end"]
    locus_scores = reference_scored[
        locus_columns + ["complexity_score", "complexity_category"]
    ].rename(
        columns={
            "complexity_score": "locus_complexity_score",
            "complexity_category": "locus_complexity_category",
        }
    )
    output = observed_scored.merge(
        locus_scores, on=locus_columns, how="left", validate="many_to_one"
    )
    if output["locus_complexity_category"].isna().any():
        raise ValueError("some contexts lack a matched strict/core locus definition")
    strict_exposure = reference[
        locus_columns + ["node_count", "unique_edge_count"]
    ].rename(
        columns={
            "node_count": "reference_context_node_count",
            "unique_edge_count": "reference_context_unique_edge_count",
        }
    )
    output = output.merge(
        strict_exposure, on=locus_columns, how="left", validate="many_to_one"
    )
    output["node_exposure_ratio_to_reference_context"] = (
        output["node_count"]
        / output["reference_context_node_count"].replace(0, pd.NA)
    )
    output["edge_exposure_ratio_to_reference_context"] = (
        output["unique_edge_count"]
        / output["reference_context_unique_edge_count"].replace(0, pd.NA)
    )

    frozen.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "manifest": str(args.manifest),
            "manifest_sha256": sha256(args.manifest),
            "dataset": args.dataset,
            "contexts_requested": args.contexts,
            "config": str(args.config),
            "config_sha256": sha256(args.config),
            "primary_category_column": "locus_complexity_category",
            "primary_score_column": "locus_complexity_score",
            "context_policy": "Fit and classify strict/core loci, then join the frozen locus label to every matched context.",
        }
    )

    output.to_csv(out_dir / "complexity_features.tsv", sep="\t", index=False)
    output.to_parquet(out_dir / "complexity_features.parquet", index=False)
    (out_dir / "complexity_thresholds.json").write_text(
        json.dumps(frozen, indent=2) + "\n", encoding="utf-8"
    )
    definition_rows = [
        {"feature": name, **description}
        for name, description in FEATURE_DEFINITIONS.items()
    ]
    pd.DataFrame(definition_rows).to_csv(
        out_dir / "complexity_feature_definitions.tsv", sep="\t", index=False
    )
    audit = {
        "status": "PASS",
        "rows": int(len(output)),
        "reference_rows": int(len(reference)),
        "contexts": output["context"].value_counts().sort_index().to_dict(),
        "locus_categories": (
            output.loc[output["context"].eq(reference_context), "locus_complexity_category"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "integrity_failures": {
            "duplicate_segment_ids": int(output["duplicate_segment_ids"].sum()),
            "missing_link_endpoint_rows": int(output["missing_link_endpoint_rows"].sum()),
            "invalid_orientation_rows": int(output["invalid_orientation_rows"].sum()),
            "invalid_candidate_labels": int(output["invalid_candidate_labels"].sum()),
            "duplicate_candidate_pairs": int(
                output["duplicate_candidate_pairs"].sum()
            ),
            "reverse_equivalent_candidate_duplicates": int(
                output["reverse_equivalent_candidate_duplicates"].sum()
            ),
            "orientation_equivalent_candidate_label_conflicts": int(
                output["orientation_equivalent_candidate_label_conflicts"].sum()
            ),
        },
        "performance_columns_read": [],
    }
    if any(audit["integrity_failures"].values()):
        audit["status"] = "FAIL"
    (out_dir / "complexity_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

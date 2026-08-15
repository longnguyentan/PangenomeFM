#!/usr/bin/env python3
"""Build the HPRC chr22 comparator pilot and fail loudly on false equality."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evaluation.comparative_benchmark import (
    canonical_oriented_pair,
    candidate_partitions,
    method_fairness_row,
    oriented_reverse_complement,
    stable_example_id,
    validate_benchmark_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    relative = manifest_path.parent / path
    if relative.exists():
        return relative
    raise FileNotFoundError(f"could not resolve {value!r} from {manifest_path}")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def normalized_chromosome(target_sn: str) -> str:
    value = str(target_sn).split("#")[-1].split("|")[-1]
    return value if value.startswith("chr") else f"chr{value}"


def oriented_sequence(sequence: str, orientation: str) -> str | None:
    if sequence in {"", "*", "nan", "None"}:
        return None
    return sequence if orientation == "+" else sequence.translate(RC_TABLE)[::-1]


def fold_spec(config: dict, fold_name: str, chromosome: str) -> tuple[str, str]:
    folds = {fold["name"]: fold for fold in config["rotating_chromosome_folds"]}
    if fold_name not in folds:
        raise ValueError(f"unknown fold {fold_name!r}; available={sorted(folds)}")
    fold = folds[fold_name]
    if chromosome in fold["test"]:
        split = "test"
    elif chromosome in fold["validation"]:
        split = "validation"
    else:
        split = "train"
    return fold_name, split


def load_complexity(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, sep="\t")
    columns = [
        "slice_id",
        "locus_complexity_score",
        "locus_complexity_category",
        "observed_context_complexity_score",
        "node_count",
        "link_row_count",
        "branching_fraction",
        "cycle_rank_per_kb",
        "alternate_node_fraction",
    ]
    return frame[[column for column in columns if column in frame.columns]].copy()


def audit_source_candidates(
    source_manifest: pd.DataFrame, manifest_path: Path
) -> pd.DataFrame:
    """Audit exact/reverse duplicates and conflicting equivalence labels."""

    rows: list[dict[str, object]] = []
    for row in source_manifest.itertuples(index=False):
        candidate_path = resolve_path(str(row.edge_pred_path), manifest_path)
        candidates = pd.read_csv(candidate_path, compression="infer")
        canonical_pairs = [
            canonical_oriented_pair(u, v)
            for u, v in zip(candidates["u_oid"], candidates["v_oid"])
        ]
        work = candidates.assign(
            _canonical_u=[pair[0] for pair in canonical_pairs],
            _canonical_v=[pair[1] for pair in canonical_pairs],
        )
        label_counts = work.groupby(["_canonical_u", "_canonical_v"])[
            "label"
        ].nunique()
        rows.append(
            {
                "slice_id": str(row.slice_id),
                "target_sn": str(row.target_sn),
                "context": str(row.closure),
                "candidate_rows": int(len(work)),
                "exact_duplicate_rows": int(
                    work.duplicated(["u_oid", "v_oid", "label"]).sum()
                ),
                "reverse_equivalent_duplicate_rows": int(
                    work.duplicated(["_canonical_u", "_canonical_v"]).sum()
                ),
                "orientation_equivalent_label_conflicts": int(
                    (label_counts > 1).sum()
                ),
                "candidate_path": str(candidate_path),
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
        "--full-segments",
        type=Path,
        default=ROOT / "data/hprc/full_segments.csv",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=ROOT / "configs/server_full_multicohort_20260806.json",
    )
    parser.add_argument(
        "--comparators",
        type=Path,
        default=ROOT / "configs/comparative_benchmark/comparators_v1.yaml",
    )
    parser.add_argument(
        "--complexity-features",
        type=Path,
        default=ROOT
        / "results/complexity/graph_window_complexity_v1/complexity_features.tsv",
    )
    parser.add_argument("--chromosome", default="chr22")
    parser.add_argument("--fold", default="fold_b")
    parser.add_argument("--contexts", nargs="+", default=["strict", "1hop"])
    parser.add_argument("--split-seed", type=int, default=20260806)
    parser.add_argument(
        "--model-seeds", nargs="+", type=int, default=[42, 314159, 20260806]
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results/comparative_benchmark/pilot",
    )
    parser.add_argument("--require-exact-all", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    protected = out_dir / "run_manifest.json"
    if protected.exists() and not args.overwrite:
        raise FileExistsError(
            f"pilot already exists at {out_dir}; use --overwrite explicitly"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    experiment = json.loads(args.experiment_config.read_text(encoding="utf-8"))
    registry = yaml.safe_load(args.comparators.read_text(encoding="utf-8"))
    fold_name, chromosome_split = fold_spec(
        experiment, args.fold, args.chromosome
    )
    if chromosome_split != "test":
        raise ValueError(
            f"pilot chromosome {args.chromosome} is {chromosome_split}, not test, in {fold_name}"
        )

    source_manifest = pd.read_csv(args.manifest)
    source_candidate_audit = audit_source_candidates(source_manifest, args.manifest)
    source_conflicts = int(
        source_candidate_audit["orientation_equivalent_label_conflicts"].sum()
    )
    selected = source_manifest.loc[
        source_manifest["target_sn"].map(normalized_chromosome).eq(args.chromosome)
        & source_manifest["closure"].isin(args.contexts)
    ].copy()
    if selected.empty:
        raise ValueError("pilot selection contains no slices")
    if set(selected["closure"]) != set(args.contexts):
        raise ValueError("not every requested context is present")

    # OIDs are defined by the order of unique segment names in the full graph.
    # Reading only the name column avoids materializing the 3+ GB sequence field.
    full_names = pd.read_csv(
        args.full_segments, usecols=["name"], dtype={"name": "string"}
    )["name"].drop_duplicates().reset_index(drop=True)

    benchmark_rows: list[dict[str, object]] = []
    deepgene_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    selected_file_checksums: dict[str, str] = {}
    requested_examples = 0
    duplicate_source_examples = 0
    label_disagreements = 0

    for slice_row in selected.sort_values(
        ["start", "end", "closure"], kind="stable"
    ).itertuples(index=False):
        segments_path = resolve_path(str(slice_row.segments_path), args.manifest)
        links_path = resolve_path(str(slice_row.links_path), args.manifest)
        candidates_path = resolve_path(str(slice_row.edge_pred_path), args.manifest)
        for path in (segments_path, links_path, candidates_path):
            selected_file_checksums[str(path)] = sha256(path)

        segments = pd.read_csv(segments_path, compression="infer")
        candidates_raw = pd.read_csv(candidates_path, compression="infer").reset_index(
            names="source_candidate_row"
        )
        requested_examples += len(candidates_raw)
        canonical_pairs = [
            canonical_oriented_pair(u, v)
            for u, v in zip(candidates_raw["u_oid"], candidates_raw["v_oid"])
        ]
        candidates_raw["_canonical_u"] = [pair[0] for pair in canonical_pairs]
        candidates_raw["_canonical_v"] = [pair[1] for pair in canonical_pairs]
        per_pair_labels = candidates_raw.groupby(
            ["_canonical_u", "_canonical_v"]
        )["label"].nunique()
        disagreements_here = int((per_pair_labels > 1).sum())
        label_disagreements += disagreements_here
        if disagreements_here:
            manifest_errors_here = (
                f"{slice_row.slice_id} has {disagreements_here} endpoint pairs with conflicting labels"
            )
            failure_rows.append(
                {
                    "slice_id": slice_row.slice_id,
                    "candidate_row": -1,
                    "method": "all",
                    "reason": manifest_errors_here,
                }
            )
        duplicate_mask = candidates_raw.duplicated(
            ["_canonical_u", "_canonical_v"], keep="first"
        )
        duplicate_source_examples += int(duplicate_mask.sum())
        for duplicate in candidates_raw.loc[duplicate_mask].itertuples(index=False):
            failure_rows.append(
                {
                    "slice_id": slice_row.slice_id,
                    "candidate_row": int(duplicate.source_candidate_row),
                    "method": "all",
                    "reason": "duplicate_candidate_pair_dropped_before_split",
                }
            )
        candidates = candidates_raw.loc[~duplicate_mask].drop(
            columns=["_canonical_u", "_canonical_v"]
        ).reset_index(drop=True)
        partitions = candidate_partitions(len(candidates), args.split_seed)
        segment_by_name = segments.drop_duplicates("name").set_index("name")

        for candidate_index, candidate in candidates.iterrows():
            u_oid = int(candidate["u_oid"])
            v_oid = int(candidate["v_oid"])
            label = int(candidate["label"])
            u_index, v_index = u_oid // 2, v_oid // 2
            if u_index >= len(full_names) or v_index >= len(full_names):
                failure_rows.append(
                    {
                        "slice_id": slice_row.slice_id,
                        "candidate_row": candidate_index,
                        "method": "pangenomefm",
                        "reason": "oriented_node_id_outside_full_segment_index",
                    }
                )
                continue
            u_name, v_name = str(full_names.iloc[u_index]), str(full_names.iloc[v_index])
            if u_name not in segment_by_name.index or v_name not in segment_by_name.index:
                failure_rows.append(
                    {
                        "slice_id": slice_row.slice_id,
                        "candidate_row": candidate_index,
                        "method": "pangenomefm",
                        "reason": "candidate_endpoint_missing_from_slice_segments",
                    }
                )
                continue

            u = segment_by_name.loc[u_name]
            v = segment_by_name.loc[v_name]
            u_orientation = "+" if u_oid % 2 == 0 else "-"
            v_orientation = "+" if v_oid % 2 == 0 else "-"
            example_id = stable_example_id(
                [
                    "hprc_local_matched_benchmark",
                    slice_row.target_sn,
                    int(slice_row.start),
                    int(slice_row.end),
                    slice_row.closure,
                    *canonical_oriented_pair(u_oid, v_oid),
                    label,
                ]
            )
            rc_u, rc_v = oriented_reverse_complement(u_oid, v_oid)
            benchmark_rows.append(
                {
                    "example_id": example_id,
                    "dataset": "hprc_local_matched_benchmark",
                    "graph_release": "HPRC_R2_inventory_matched_raw_checksum_unverified",
                    "coordinate_system": "GRCh38",
                    "target_sn": str(slice_row.target_sn),
                    "chromosome": args.chromosome,
                    "start": int(slice_row.start),
                    "end": int(slice_row.end),
                    "window_bp": int(slice_row.end) - int(slice_row.start),
                    "window_id": f"{args.chromosome}:{int(slice_row.start)}-{int(slice_row.end)}",
                    "slice_id": str(slice_row.slice_id),
                    "fold": fold_name,
                    "chromosome_split": chromosome_split,
                    "candidate_partition": str(partitions[candidate_index]),
                    "split_seed": args.split_seed,
                    "model_seeds": json.dumps(args.model_seeds),
                    "context": str(slice_row.closure),
                    "context_definition": str(slice_row.context_regime),
                    "source_oid": u_oid,
                    "destination_oid": v_oid,
                    "reverse_complement_source_oid": rc_u,
                    "reverse_complement_destination_oid": rc_v,
                    "source_segment_index": u_index,
                    "destination_segment_index": v_index,
                    "source_segment_name": u_name,
                    "destination_segment_name": v_name,
                    "source_orientation": u_orientation,
                    "destination_orientation": v_orientation,
                    "source_coordinate": pd.to_numeric(u.get("SO"), errors="coerce"),
                    "destination_coordinate": pd.to_numeric(v.get("SO"), errors="coerce"),
                    "source_length_bp": pd.to_numeric(u.get("LN"), errors="coerce"),
                    "destination_length_bp": pd.to_numeric(v.get("LN"), errors="coerce"),
                    "source_path_name": str(u.get("SN", "")),
                    "destination_path_name": str(v.get("SN", "")),
                    "source_sr": pd.to_numeric(u.get("SR"), errors="coerce"),
                    "destination_sr": pd.to_numeric(v.get("SR"), errors="coerce"),
                    "source_is_reference_rank": bool(pd.to_numeric(u.get("SR"), errors="coerce") == 0),
                    "destination_is_reference_rank": bool(pd.to_numeric(v.get("SR"), errors="coerce") == 0),
                    "label": label,
                    "region_annotations": "[]",
                    "sv_annotations": "[]",
                    "path_support_status": "not_available_in_slice_tables",
                    "provenance": json.dumps(
                        {
                            "source_manifest": str(args.manifest),
                            "candidate_file": str(candidates_path),
                            "candidate_row_zero_based": int(candidate["source_candidate_row"]),
                            "segments_file": str(segments_path),
                            "links_file": str(links_path),
                        },
                        sort_keys=True,
                    ),
                }
            )

            u_sequence = oriented_sequence(str(u.get("seq", "")), u_orientation)
            v_sequence = oriented_sequence(str(v.get("seq", "")), v_orientation)
            convertible = u_sequence is not None and v_sequence is not None
            deepgene_rows.append(
                {
                    "example_id": example_id,
                    "adapter_status": (
                        "approximately_convertible" if convertible else "missing_sequence"
                    ),
                    "adapter_task": "supervised_oriented_endpoint_sequence_pair_classification",
                    "source_sequence_length": len(u_sequence) if u_sequence else np.nan,
                    "destination_sequence_length": len(v_sequence) if v_sequence else np.nan,
                    "source_sequence_sha256": (
                        hashlib.sha256(u_sequence.upper().encode()).hexdigest()
                        if u_sequence
                        else ""
                    ),
                    "destination_sequence_sha256": (
                        hashlib.sha256(v_sequence.upper().encode()).hexdigest()
                        if v_sequence
                        else ""
                    ),
                    "label": label,
                    "scientific_warning": "Not the published DeepGene task; requires a new supervised head and frozen training protocol.",
                }
            )

    benchmark = pd.DataFrame(benchmark_rows)
    deepgene = pd.DataFrame(deepgene_rows)
    complexity = load_complexity(args.complexity_features)
    if not complexity.empty:
        benchmark = benchmark.merge(complexity, on="slice_id", how="left", validate="many_to_one")
    else:
        benchmark["locus_complexity_score"] = np.nan
        benchmark["locus_complexity_category"] = "not_computed"

    manifest_errors = validate_benchmark_manifest(benchmark)
    if label_disagreements:
        manifest_errors.append(
            f"{label_disagreements} endpoint pairs have conflicting labels"
        )
    expected_unique_examples = requested_examples - duplicate_source_examples
    if expected_unique_examples != len(benchmark):
        manifest_errors.append(
            f"expected {expected_unique_examples} unique candidate rows after deduplication but materialized {len(benchmark)}"
        )

    methods = registry["methods"]
    all_ids = benchmark["example_id"].astype(str).tolist()
    deepgene_approx_ids = deepgene.loc[
        deepgene["adapter_status"].eq("approximately_convertible"), "example_id"
    ].astype(str).tolist()
    audit_rows = [
        method_fairness_row(
            method="PangenomeFM",
            compatibility=methods["pangenomefm"]["compatibility"],
            requested_examples=requested_examples,
            exact_example_ids=all_ids,
            approximate_example_ids=all_ids,
            benchmark=benchmark,
            status=(
                "PILOT_MANIFEST_READY_FULL_FOLD_BLOCKED_SOURCE_LABEL_CONFLICTS"
                if source_conflicts and not manifest_errors
                else ("READY_FOR_EXECUTION" if not manifest_errors else "FAIL")
            ),
            blocker=(
                f"The selected chr22 manifest is valid, but full fold-b training is blocked by {source_conflicts} orientation-equivalent label conflicts elsewhere in the local source benchmark. Regenerate candidates; do not resolve labels silently."
                if source_conflicts
                else "Exact native task; checkpoint execution remains to be scheduled."
            ),
            duplicate_source_examples=duplicate_source_examples,
        ),
        method_fairness_row(
            method="DeepGene",
            compatibility=methods["deepgene"]["compatibility"],
            requested_examples=requested_examples,
            exact_example_ids=[],
            approximate_example_ids=deepgene_approx_ids,
            benchmark=benchmark,
            status="BLOCKED_EXACT_APPROXIMATE_ADAPTER_PREPARED",
            blocker=methods["deepgene"]["exact_edge_task_blocker"],
            duplicate_source_examples=duplicate_source_examples,
        ),
        method_fairness_row(
            method="PangenomeX",
            compatibility=methods["pangenomex"]["compatibility"],
            requested_examples=requested_examples,
            exact_example_ids=[],
            approximate_example_ids=[],
            benchmark=benchmark,
            status="CONTEXTUAL_ONLY",
            blocker=methods["pangenomex"]["exact_edge_task_blocker"],
            duplicate_source_examples=duplicate_source_examples,
        ),
    ]
    audit = pd.DataFrame(audit_rows)
    failures = pd.DataFrame(
        failure_rows,
        columns=["slice_id", "candidate_row", "method", "reason"],
    )

    benchmark.to_csv(out_dir / "benchmark_manifest.tsv", sep="\t", index=False)
    benchmark.to_parquet(out_dir / "benchmark_manifest.parquet", index=False)
    benchmark.to_csv(
        out_dir / "pangenomefm_examples.tsv", sep="\t", index=False
    )
    deepgene.to_csv(
        out_dir / "deepgene_approximate_adapter_manifest.tsv", sep="\t", index=False
    )
    failures.to_csv(out_dir / "conversion_failures.tsv", sep="\t", index=False)
    selected.to_csv(out_dir / "source_slice_manifest.csv", index=False)
    source_candidate_audit.to_csv(
        out_dir / "source_benchmark_candidate_audit.tsv", sep="\t", index=False
    )
    audit.to_csv(out_dir / "benchmark_audit.tsv", sep="\t", index=False)
    audit_payload = {
        "status": "FAIL" if manifest_errors else "PASS_WITH_TASK_MISMATCHES",
        "manifest_errors": manifest_errors,
        "methods": audit.to_dict(orient="records"),
        "drop_reasons": dict(Counter(failures.get("reason", []))),
        "source_benchmark_candidate_audit": {
            "slices": int(len(source_candidate_audit)),
            "candidate_rows": int(source_candidate_audit["candidate_rows"].sum()),
            "exact_duplicate_rows": int(
                source_candidate_audit["exact_duplicate_rows"].sum()
            ),
            "reverse_equivalent_duplicate_rows": int(
                source_candidate_audit["reverse_equivalent_duplicate_rows"].sum()
            ),
            "orientation_equivalent_label_conflicts": source_conflicts,
            "conflict_slices": int(
                (source_candidate_audit["orientation_equivalent_label_conflicts"] > 0).sum()
            ),
            "execution_gate": (
                "BLOCK_FULL_FOLD_REGENERATE_CANDIDATES"
                if source_conflicts
                else "PASS"
            ),
        },
        "interpretation": "Only PangenomeFM is exact for masked endpoint relation reconstruction. DeepGene is approximate and PangenomeX is contextual; do not rank all three in one performance table.",
    }
    (out_dir / "benchmark_audit.json").write_text(
        json.dumps(audit_payload, indent=2) + "\n", encoding="utf-8"
    )

    smoke_path = out_dir / "pangenomefm_smoke_summary.json"
    smoke = (
        json.loads(smoke_path.read_text(encoding="utf-8"))
        if smoke_path.exists()
        else {}
    )
    metric_template = pd.DataFrame(
        [
            {
                "method": name,
                "comparison_tier": methods[key]["compatibility"],
                "examples": (
                    len(benchmark)
                    if key == "pangenomefm"
                    else (len(deepgene_approx_ids) if key == "deepgene" else 0)
                ),
                "AUROC": np.nan,
                "AUPRC": np.nan,
                "runtime_seconds": (
                    smoke.get("measured_resources", {}).get("wall_seconds")
                    if key == "pangenomefm" and smoke
                    else np.nan
                ),
                "peak_ram_gib": (
                    smoke.get("measured_resources", {}).get("peak_ram_gib")
                    if key == "pangenomefm" and smoke
                    else np.nan
                ),
                "peak_gpu_memory_gib": np.nan,
                "status": audit_rows[index]["status"],
                "notes": (
                    audit_rows[index]["blocker_or_note"]
                    + (
                        " A deterministic one-epoch chr22 inner-split code-path smoke passed; its score is non-promotable."
                        if key == "pangenomefm" and smoke
                        else ""
                    )
                ),
            }
            for index, (key, name) in enumerate(
                [
                    ("pangenomefm", "PangenomeFM"),
                    ("deepgene", "DeepGene"),
                    ("pangenomex", "PangenomeX"),
                ]
            )
        ]
    )
    metric_template.to_csv(out_dir / "pilot_method_status.tsv", sep="\t", index=False)

    reproducibility = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "dataset": "hprc_local_matched_benchmark",
        "graph_release": "HPRC_R2_inventory_matched_raw_checksum_unverified",
        "coordinate_system": "GRCh38",
        "chromosome": args.chromosome,
        "fold": fold_name,
        "chromosome_split": chromosome_split,
        "contexts": args.contexts,
        "split_seed": args.split_seed,
        "model_seeds": args.model_seeds,
        "inputs": {
            str(args.manifest): sha256(args.manifest),
            str(args.full_segments): sha256(args.full_segments),
            str(args.experiment_config): sha256(args.experiment_config),
            str(args.comparators): sha256(args.comparators),
            **selected_file_checksums,
        },
        "outputs": [
            "benchmark_manifest.tsv",
            "benchmark_manifest.parquet",
            "benchmark_audit.tsv",
            "benchmark_audit.json",
            "conversion_failures.tsv",
            "deepgene_approximate_adapter_manifest.tsv",
            "pangenomefm_examples.tsv",
            "pilot_method_status.tsv",
            "source_slice_manifest.csv",
            "source_benchmark_candidate_audit.tsv",
            *( ["pangenomefm_smoke_summary.json"] if smoke else [] ),
        ],
        "completion_status": audit_payload["status"],
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(reproducibility, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(audit_payload, indent=2))
    if manifest_errors:
        raise SystemExit(2)
    if args.require_exact_all and any(
        method["compatibility"] != "exact" for method in methods.values()
    ):
        print("ERROR: --require-exact-all failed because one or more methods are not exact", file=sys.stderr)
        raise SystemExit(3)


if __name__ == "__main__":
    main()

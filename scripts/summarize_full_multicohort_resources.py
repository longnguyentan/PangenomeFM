#!/usr/bin/env python3
"""Summarize verified files, graph validation, paths, and benchmark coverage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REFERENCES = {"CHM13", "GRCh38", "GRCh37"}


def _expand_environment(value):
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "${" in expanded:
            raise ValueError(f"Unresolved environment variable: {value}")
        return expanded
    return value


def _file(path_value: str | None) -> dict[str, Any]:
    if not path_value:
        return {"path": None, "exists": False, "bytes": None}
    path = ROOT / path_value
    return {
        "path": path_value,
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
    }


def _path_inventory(path_value: str | None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path_metadata_records": None,
        "path_samples_excluding_references": None,
        "path_haplotypes_excluding_references": None,
        "path_loci_or_contigs": None,
    }
    if not path_value or not (ROOT / path_value).exists():
        return info
    path = ROOT / path_value
    separator = "\t" if path.suffix == ".tsv" else ","
    frame = pd.read_csv(path, sep=separator, compression="infer")
    if "SAMPLE" in frame:
        sample, haplotype = "SAMPLE", "HAPLOTYPE"
        selected = frame[~frame[sample].astype(str).isin(REFERENCES)]
        locus = "LOCUS"
    else:
        sample, haplotype = "sample", "haplotype"
        selected = frame[~frame[sample].astype(str).isin(REFERENCES)]
        locus = "contig"
    info.update(
        {
            "path_metadata_records": int(len(frame)),
            "path_samples_excluding_references": int(selected[sample].nunique()),
            "path_haplotypes_excluding_references": int(
                selected[[sample, haplotype]].drop_duplicates().shape[0]
            ),
            "path_loci_or_contigs": int(frame[locus].nunique()),
        }
    )
    if "step_count" in frame:
        info["path_steps_indexed"] = int(frame["step_count"].sum())
    return info


def _benchmark_inventory(path_value: str | None) -> list[dict[str, Any]]:
    if not path_value:
        return []
    manifest = ROOT / path_value / "manifest.csv"
    if not manifest.exists():
        return []
    frame = pd.read_csv(manifest)
    rows: list[dict[str, Any]] = []
    for closure, group in frame.groupby("closure", sort=True):
        rows.append(
            {
                "benchmark": path_value,
                "closure": str(closure),
                "slices": int(len(group)),
                "chromosomes": int(group["target_sn"].nunique()),
                "visible_nodes_sum": int(group["n_segments"].sum()),
                "visible_links_sum": int(group["n_links"].sum()),
                "coordinate_bp_sum": int((group["end"] - group["start"]).sum()),
                "negative_sampler": (
                    str(group["negative_sampler"].iloc[0])
                    if "negative_sampler" in group
                    else None
                ),
                "degree_matched": (
                    bool(group["negative_degree_matched"].all())
                    if "negative_degree_matched" in group
                    else None
                ),
                "tile_stride_bp": (
                    int(group["tile_stride_bp"].dropna().iloc[0])
                    if "tile_stride_bp" in group
                    and not group["tile_stride_bp"].dropna().empty
                    else None
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/full_multicohort_all_chromosomes_20260806.json",
    )
    parser.add_argument(
        "--out-dir",
        default="results/full_multicohort_all_chromosomes_20260806/resource_inventory",
    )
    args = parser.parse_args()
    config = _expand_environment(
        json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    )
    out_dir = Path(os.path.expandvars(args.out_dir))
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    resource_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    for name, dataset in config["datasets"].items():
        validation_path = (
            ROOT
            / config["outputs"]["validation"]
            / name
            / "validation_summary.json"
        )
        validation = (
            json.loads(validation_path.read_text(encoding="utf-8"))
            if validation_path.exists()
            else {}
        )
        files = {
            key: _file(dataset.get(key))
            for key in ["raw_gfa", "raw_gbz", "segments", "links", "path_metadata"]
        }
        row: dict[str, Any] = {
            "dataset": name,
            "release": dataset["release"],
            "role": dataset["role"],
            "source_format": dataset.get("source_format"),
            "declared_graph_donors": dataset.get("graph_donors"),
            "declared_graph_haplotypes": dataset.get(
                "graph_haplotypes",
                dataset.get("graph_haplotypes_total_including_references"),
            ),
            "sequence_representation": dataset.get(
                "sequence_representation", "GFA S-record nucleotide sequence"
            ),
            "variant_representation": dataset.get(
                "variant_representation", "SV-scale graph topology"
            ),
            "functional_annotations": dataset.get(
                "functional_annotations", "not embedded in graph resource"
            ),
            "data_use_notes": dataset.get("data_use_notes"),
            "reference_prefix": dataset["reference_prefix"],
            "validation_status": validation.get("validation_status", "not_yet_executed"),
            "segments": validation.get("segments"),
            "links": validation.get("links"),
            "graph_sequence_names": validation.get("unique_sequence_names"),
            "graph_donors_from_sn": validation.get("donors"),
            "graph_haplotypes_from_sn": validation.get("phased_haplotypes"),
            "components": validation.get("components"),
            "isolated_nodes": validation.get("isolated_nodes"),
            "fatal_issue_count": validation.get("fatal_issue_count"),
            **_path_inventory(dataset.get("path_metadata")),
        }
        for key, info in files.items():
            row[f"{key}_path"] = info["path"]
            row[f"{key}_exists"] = info["exists"]
            row[f"{key}_bytes"] = info["bytes"]
        resource_rows.append(row)
        for kind in ["benchmark", "pretrain_benchmark"]:
            for benchmark_row in _benchmark_inventory(dataset.get(kind)):
                benchmark_rows.append(
                    {"dataset": name, "benchmark_kind": kind, **benchmark_row}
                )

    resources = pd.DataFrame(resource_rows)
    benchmarks = pd.DataFrame(benchmark_rows)
    resources.to_csv(out_dir / "dataset_manifest.csv", index=False)
    benchmarks.to_csv(out_dir / "benchmark_coverage.csv", index=False)

    decisions = pd.DataFrame(
        [
            {
                "resource": "HPRC Release 2",
                "decision": "primary_training",
                "reason": "Latest stable integrated HPRC graph locally available.",
            },
            {
                "resource": "HGSVC3",
                "decision": "primary_training_and_independent_transfer",
                "reason": "Compatible complete SV graph and independent 65-donor cohort.",
            },
            {
                "resource": "HPRC Release 1.1",
                "decision": "release_transfer_only",
                "reason": "Pooling with R2 duplicates superseded individuals and risks leakage.",
            },
            {
                "resource": "Official HGSVC3 + HPRC-v1 integrated graph",
                "decision": "separate_integrated_graph_stress_test",
                "reason": "Versioned 216-haplotype release, but it contains HPRC-v1 rather than R2 and must not be counted as an additional independent cohort.",
            },
            {
                "resource": "HPRC Release 3",
                "decision": "inventory_only",
                "reason": config["release_3_policy"]["reason"],
            },
            {
                "resource": "Other public human pangenomes",
                "decision": "not_locally_available_or_versioned_in_repository",
                "reason": "No further stable, versioned, format-compatible integrated human graph was identified in the pinned public-resource audit.",
            },
        ]
    )
    decisions.to_csv(out_dir / "resource_inclusion_decisions.csv", index=False)
    summary = {
        "analysis_id": config["analysis_id"],
        "datasets": int(len(resources)),
        "validated_datasets": int(
            (resources["validation_status"] != "not_yet_executed").sum()
        ),
        "benchmark_rows": int(len(benchmarks)),
        "combined_unique_donors": config["combined_unique_graph_cohort"][
            "unique_donors"
        ],
        "primary_chromosomes_requested": config["primary_chromosomes"],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(resources.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

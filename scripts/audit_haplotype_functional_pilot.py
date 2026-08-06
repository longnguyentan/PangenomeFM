"""Audit whether local data are sufficient for haplotype-functional pilots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parse_summaries = sorted(data_root.glob("hgsvc*/**/parse_summary.json"))
    path_summaries = sorted(data_root.glob("hgsvc*/**/path_summary.json"))
    path_index_summaries = sorted(
        data_root.glob("hgsvc*/**/path_index_summary.json")
    )
    gbz_path_tables = sorted(data_root.glob("public/hprc/**/paths.metadata.tsv"))
    gbz_path_records = 0
    for path in gbz_path_tables:
        with path.open("r", encoding="utf-8") as handle:
            gbz_path_records += max(sum(1 for _ in handle) - 1, 0)
    path_records = sum(
        int(_json(path).get("paths", 0)) + int(_json(path).get("walks", 0))
        for path in parse_summaries
    )
    extracted_steps = sum(int(_json(path).get("path_steps", 0)) for path in path_summaries)
    indexed_records = sum(
        int(_json(path).get("path_records", 0))
        + int(_json(path).get("walk_records", 0))
        for path in path_index_summaries
    )

    suffixes = {
        "aligned_rna": {".bam", ".cram"},
        "raw_rna": {".fastq", ".fq"},
        "functional_tracks": {".bigwig", ".bw", ".bed"},
        "variant_or_ase": {".vcf", ".bcf", ".tsv"},
        "methylation": {".methylc"},
    }
    files: dict[str, list[str]] = {key: [] for key in suffixes}
    for path in data_root.glob("**/*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        for category, category_suffixes in suffixes.items():
            if any(
                name.endswith(suffix)
                or name.endswith(f"{suffix}.gz")
                for suffix in category_suffixes
            ):
                files[category].append(str(path))

    methylation_summaries = sorted(
        Path("results").glob("**/haplotype_methylation*/dataset_summary.json")
    )
    completed_methylation = [
        {"summary_path": str(path), **_json(path)}
        for path in methylation_summaries
        if int(_json(path).get("n_reciprocal_h1_h2_pairs", 0)) > 0
        and _json(path).get("path_names")
    ]
    completed_methylation_donors = sorted(
        {
            str(sample)
            for summary in completed_methylation
            for sample in summary.get("sample_ids", [])
        }
    )
    donor_benchmark_paths = sorted(
        Path("results").glob(
            "**/haplotype_methylation*donor*/summary.json"
        )
    )
    completed_donor_benchmarks = [
        {"summary_path": str(path), **_json(path)}
        for path in donor_benchmark_paths
        if _json(path).get("split_mode") == "donor"
    ]
    expression_summaries = sorted(
        Path("results").glob("**/haplotype_expression*/dataset_summary.json")
    )
    completed_expression = [
        {"summary_path": str(path), **_json(path)}
        for path in expression_summaries
        if int(_json(path).get("n_reciprocal_h1_h2_pairs", 0)) > 0
    ]

    has_paths = (
        path_records > 0
        or indexed_records > 0
        or extracted_steps > 0
        or gbz_path_records > 0
    )
    has_expression = bool(files["aligned_rna"] or files["raw_rna"])
    has_haplotype_expression_tracks = any(
        "hap1" in Path(path).name.lower() or "hap2" in Path(path).name.lower()
        for path in files["functional_tracks"]
    )
    has_ase_labels = any("ase" in Path(path).name.lower() for path in files["variant_or_ase"])
    has_methylation = bool(files["methylation"])
    ase_ready = has_paths and has_expression and has_ase_labels
    methylation_ready = has_paths and has_methylation
    methylation_completed = bool(completed_methylation)
    ase_blockers = []
    if not has_paths:
        ase_blockers.append("No indexed P/W haplotype path records are available.")
    if not has_expression:
        ase_blockers.append("No donor-matched RNA-seq FASTQ/BAM/CRAM is present locally.")
    if not has_ase_labels:
        ase_blockers.append("No phased allele-specific expression count table is present locally.")
    methylation_blockers = []
    if not has_paths:
        methylation_blockers.append("No indexed P/W haplotype path records are available.")
    if not has_methylation:
        methylation_blockers.append("No haplotype-resolved methylation track is present locally.")

    summary = {
        "pilots": {
            "hprc_haplotype_methylation": {
                "ready_to_build": methylation_ready,
                "completed_feasibility_run": methylation_completed,
                "completed_summaries": completed_methylation,
                "completed_donors": completed_methylation_donors,
                "heldout_donor_benchmarks": completed_donor_benchmarks,
                "blockers": methylation_blockers,
                "implemented_runner": "python -m tasks.haplotype.methylation",
            },
            "hgsvc3_or_geuvadis_ase": {
                "ready_to_train": ase_ready,
                "blockers": ase_blockers,
                "implemented_runner": "python -m tasks.haplotype.ase",
            },
            "hprc_haplotype_expression_track": {
                "tracks_acquired": has_haplotype_expression_tracks,
                "completed_feasibility_run": bool(completed_expression),
                "completed_summaries": completed_expression,
                "status": (
                    "processed haplotype tracks available for a direct-signal "
                    "pilot; gene-level ASE still requires gene models and "
                    "held-out donors"
                    if has_haplotype_expression_tracks
                    else "no haplotype-separated expression tracks found"
                ),
            },
        },
        "path_records_reported": path_records,
        "path_records_indexed": indexed_records,
        "gbz_path_records": gbz_path_records,
        "extracted_path_steps": extracted_steps,
        "parse_summaries": [str(path) for path in parse_summaries],
        "path_summaries": [str(path) for path in path_summaries],
        "path_index_summaries": [str(path) for path in path_index_summaries],
        "gbz_path_tables": [str(path) for path in gbz_path_tables],
        "candidate_files": files,
        "required_pair_schema": [
            "donor_id",
            "h1_embedding_id",
            "h2_embedding_id",
            "ase_value OR h1_count+h2_count",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Haplotype-specific functional pilot audit",
        "",
        f"Methylation pilot ready: **{'yes' if methylation_ready else 'no'}**",
        f"Methylation feasibility run complete: **{'yes' if methylation_completed else 'no'}**",
        f"Methylation donors completed: **{len(completed_methylation_donors)}**",
        f"Leave-one-donor-out benchmark complete: **{'yes' if completed_donor_benchmarks else 'no'}**",
        f"ASE pilot ready: **{'yes' if ase_ready else 'no'}**",
        "",
        f"- GFA P/W records reported: {path_records}",
        f"- Compactly indexed P/W records: {indexed_records}",
        f"- GBZ path metadata records: {gbz_path_records}",
        f"- Extracted path steps: {extracted_steps}",
        f"- Haplotype methylation tracks: {len(files['methylation'])}",
        f"- Functional BigWig/BED tracks: {len(files['functional_tracks'])}",
        f"- RNA alignment/read files: {len(files['aligned_rna']) + len(files['raw_rna'])}",
        f"- ASE-named tables: {sum('ase' in Path(p).name.lower() for p in files['variant_or_ase'])}",
        "",
        "## ASE blocking inputs",
        "",
        *[f"- {blocker}" for blocker in ase_blockers],
        "",
        "The methylation pilot now includes a two-donor leave-one-donor-out "
        "stress test. Two donors cannot support stable donor-level confidence "
        "intervals; a paper-level headline still requires more donors and a "
        "stronger sequence baseline.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

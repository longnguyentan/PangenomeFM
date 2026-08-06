"""Build a local, machine-readable catalog of public haplotype resources."""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pandas as pd


SAMPLE_RE = re.compile(r"(?<![A-Z0-9])(?:HG|NA)\d{5}(?![A-Z0-9])")
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def _sample_token(value: str) -> str | None:
    value = str(value).removeprefix("id=")
    value = value.split("|", 1)[0].split("#", 1)[0]
    value = re.sub(r"\.[12]$", "", value)
    if value in {"CHM13", "GRCh38", "GRCh37", ""}:
        return None
    match = SAMPLE_RE.search(value)
    return match.group(0) if match else value


def _segment_samples(path: Path) -> set[str]:
    values = pd.read_csv(path, usecols=["SN"], compression="infer")["SN"].dropna()
    return {sample for value in values.unique() if (sample := _sample_token(value))}


def _s3_prefix_samples(path: Path) -> set[str]:
    root = ET.parse(path).getroot()
    samples: set[str] = set()
    for node in root.findall(".//s3:CommonPrefixes/s3:Prefix", S3_NS):
        if node.text:
            sample = node.text.rstrip("/").split("/")[-1]
            if sample:
                samples.add(sample)
    return samples


def _json_sample_ids(path: Path) -> set[str]:
    return set(SAMPLE_RE.findall(path.read_text(encoding="utf-8")))


def _entex_assays(path: Path) -> tuple[pd.DataFrame, int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    records = value.get("@graph", [])
    counts = Counter(str(record.get("assay_title", "unknown")) for record in records)
    rows = [{"assay": assay, "n_experiments": count} for assay, count in counts.most_common()]
    return pd.DataFrame(rows), int(value.get("total", len(records)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/resource_catalog/raw")
    parser.add_argument("--out-dir", default="data/resource_catalog")
    parser.add_argument("--hprc-segments", default="data/hprc/full_segments.csv")
    parser.add_argument(
        "--hgsvc-segments",
        default="data/hgsvc3_expanded/full_segments.csv.gz",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hprc_graph = _segment_samples(Path(args.hprc_segments))
    hgsvc_graph = _segment_samples(Path(args.hgsvc_segments))
    hprc_epigenome = _s3_prefix_samples(raw_dir / "hprc_epigenome_samples.xml")
    geuvadis = _json_sample_ids(raw_dir / "geuvadis_E-GEUV-3.json")
    entex_assays, entex_total = _entex_assays(raw_dir / "entex_experiments.json")

    universes = {
        "hprc_graph": hprc_graph,
        "hgsvc_graph": hgsvc_graph,
        "hprc_epigenome": hprc_epigenome,
        "geuvadis": geuvadis,
    }
    all_samples = sorted(set().union(*universes.values()))
    overlap_rows = []
    for sample in all_samples:
        overlap_rows.append(
            {"sample": sample, **{name: sample in values for name, values in universes.items()}}
        )
    overlaps = pd.DataFrame(overlap_rows)
    overlaps.to_csv(out_dir / "donor_overlaps.csv", index=False)
    entex_assays.to_csv(out_dir / "entex_assay_counts.csv", index=False)

    summary = {
        "resource_counts": {name: len(values) for name, values in universes.items()},
        "overlaps": {
            "hprc_graph_and_hgsvc_graph": len(hprc_graph & hgsvc_graph),
            "hprc_graph_and_hprc_epigenome": len(hprc_graph & hprc_epigenome),
            "hgsvc_graph_and_geuvadis": len(hgsvc_graph & geuvadis),
            "hprc_graph_and_geuvadis": len(hprc_graph & geuvadis),
        },
        "overlap_samples": {
            "hprc_graph_and_hgsvc_graph": sorted(hprc_graph & hgsvc_graph),
            "hgsvc_graph_and_geuvadis": sorted(hgsvc_graph & geuvadis),
        },
        "entex_experiments": entex_total,
        "selected_first_pilot": {
            "resource": "HPRC Release 2 epigenome",
            "reason": "Largest direct donor overlap with the current HPRC graph and open hap1/hap2 tracks",
        },
    }
    (out_dir / "catalog_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# Public resource feasibility catalog",
        "",
        "## Sample counts",
        "",
        *[f"- {name}: {len(values)}" for name, values in universes.items()],
        "",
        "## Direct overlaps",
        "",
        f"- HPRC graph ∩ HPRC epigenome: {len(hprc_graph & hprc_epigenome)}",
        f"- HGSVC graph ∩ GEUVADIS: {len(hgsvc_graph & geuvadis)}",
        f"- HPRC graph ∩ GEUVADIS: {len(hprc_graph & geuvadis)}",
        f"- HPRC graph ∩ HGSVC graph: {len(hprc_graph & hgsvc_graph)}",
        "",
        "The HPRC epigenome is the first-choice functional pilot because it has the",
        "largest direct donor overlap and already exposes haplotype-separated tracks.",
        "GEUVADIS is the second-choice expression pilot because it supplies a larger",
        "held-out-donor cohort but requires ASE generation and mapping-bias controls.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

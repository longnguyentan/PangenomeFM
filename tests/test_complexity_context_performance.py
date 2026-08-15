from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.server.analyze_complexity_context_performance import (
    chromosome_block_summary,
    main,
    paired_context_rows,
)


def test_chromosome_bootstrap_weights_chromosomes_equally() -> None:
    frame = pd.DataFrame(
        {
            "chromosome": ["chr1"] * 10 + ["chr2"],
            "value": [1.0] * 10 + [3.0],
        }
    )
    result = chromosome_block_summary(
        frame,
        "value",
        n_bootstrap=200,
        rng=np.random.default_rng(1),
    )
    assert result["n_chromosomes"] == 2
    assert result["chromosome_block_mean"] == 2.0


def test_paired_context_uses_same_locus_and_baseline() -> None:
    frame = pd.DataFrame(
        {
            "baseline": ["degree"] * 2,
            "chromosome": ["chr1"] * 2,
            "start": [0, 0],
            "end": [10, 10],
            "fold": ["fold_a"] * 2,
            "locus_complexity_category": ["low"] * 2,
            "locus_complexity_score": [0.1] * 2,
            "context": ["strict", "1hop"],
            "auprc_advantage": [0.1, 0.4],
            "model_score": [0.2, 0.5],
            "auroc_advantage": [0.0, 0.1],
        }
    )
    paired = paired_context_rows(frame)
    assert len(paired) == 1
    assert paired.loc[0, "auprc_advantage__expanded_minus_strict"] == pytest.approx(0.3)
    assert paired.loc[0, "model_score__expanded_minus_strict"] == pytest.approx(0.3)


def test_command_line_analysis_writes_audited_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    complexity_dir = tmp_path / "complexity"
    complexity_dir.mkdir()
    complexity_rows = []
    score_rows = {"strict": [], "1hop": []}
    categories = ["low", "medium", "high"]
    for chromosome_index, chromosome in enumerate(["chr1", "chr2"]):
        for category_index, category in enumerate(categories):
            start = category_index * 10
            for context in ["strict", "1hop"]:
                region_id = f"{chromosome}_{start}_{context}"
                complexity_rows.append(
                    {
                        "slice_id": region_id,
                        "chromosome": chromosome,
                        "start": start,
                        "end": start + 10,
                        "context": context,
                        "locus_complexity_score": float(category_index),
                        "locus_complexity_category": category,
                        "observed_context_complexity_score": float(category_index),
                        "node_exposure_ratio_to_reference_context": (
                            1.0 if context == "strict" else 1.5
                        ),
                        "edge_exposure_ratio_to_reference_context": (
                            1.0 if context == "strict" else 1.8
                        ),
                    }
                )
                score_rows[context].append(
                    {
                        "region_id": region_id,
                        "chromosome": chromosome,
                        "start": start,
                        "end": start + 10,
                        "fold": f"fold_{chromosome_index}",
                        "baseline": "topology_degree_sum",
                        "context": context,
                        "is_eligible": True,
                        "model_score": 0.1 + 0.05 * category_index,
                        "auprc_advantage": 0.02 + 0.01 * category_index,
                        "auroc_advantage": 0.01 + 0.01 * category_index,
                    }
                )
    complexity_path = complexity_dir / "complexity_features.tsv"
    pd.DataFrame(complexity_rows).to_csv(complexity_path, sep="\t", index=False)
    (complexity_dir / "complexity_audit.json").write_text(
        json.dumps({"status": "PASS", "performance_columns_read": []})
    )

    score_specs = []
    for context, rows in score_rows.items():
        score_dir = tmp_path / context
        score_dir.mkdir()
        score_path = score_dir / "region_scores.csv"
        pd.DataFrame(rows).to_csv(score_path, index=False)
        (score_dir / "audit.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "downstream_signal_access": "none; synthetic test",
                }
            )
        )
        score_specs.extend(["--score", f"topology_degree_sum={score_path}"])

    output = tmp_path / "analysis"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_complexity_context_performance.py",
            "--complexity",
            str(complexity_path),
            *score_specs,
            "--out-dir",
            str(output),
            "--n-bootstrap",
            "100",
            "--n-permutations",
            "100",
        ],
    )
    assert main() == 0
    assert (output / "audit.json").is_file()
    assert (output / "figure4_complexity_context.png").stat().st_size > 0
    paired = pd.read_csv(output / "paired_context_region_deltas.csv")
    assert len(paired) == 6
    summary = pd.read_csv(output / "complexity_stratum_summary.csv")
    assert set(summary["metric"]) == {
        "auprc_advantage",
        "model_score",
        "auroc_advantage",
    }

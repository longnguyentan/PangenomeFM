#!/usr/bin/env python3
"""Summarize the completed fold-A frozen/random control matrix.

The control runner intentionally keeps the original probe outputs separate from
the v2 review.  This script only reads compact ``metrics.csv`` files; it never
refits a probe or selects a model.  Rows produced from the random checkpoint are
named ``random`` even though the legacy probe calls that column ``T``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _feature_names(task: str) -> dict[str, str]:
    suffix = "_pair" if task == "sv" else ""
    return {
        "cs": f"coordinate_plus_frozen_sequence_fm{suffix}",
        "csh": f"coordinate_plus_frozen_sequence_fm_plus_topology_control{suffix}",
        "cst": f"coordinate_plus_frozen_sequence_fm_plus_frozen_pangenomefm{suffix}",
        "csht": (
            f"coordinate_plus_frozen_sequence_fm_plus_topology_control_plus_"
            f"frozen_pangenomefm{suffix}"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frames: list[pd.DataFrame] = []
    for model in ("v1", "random"):
        for task in ("sv", "ccre"):
            for context in ("strict", "1hop"):
                path = args.input_root / model / task / context / "metrics.csv"
                if not path.exists():
                    raise FileNotFoundError(path)
                frame = pd.read_csv(path)
                frame.insert(0, "model", model)
                frame.insert(1, "task", task)
                frame.insert(2, "context", context)
                frames.append(frame)

    metrics = pd.concat(frames, ignore_index=True)
    keys = ["model", "task", "context"]
    rows: list[dict[str, object]] = []
    for key, group in metrics.groupby(keys, sort=True):
        model, task, context = key
        names = _feature_names(task)
        indexed = group.set_index("feature_set")
        missing = [name for name in names.values() if name not in indexed.index]
        if missing:
            raise ValueError(f"{key}: missing feature rows {missing}")
        row: dict[str, object] = {"model": model, "task": task, "context": context}
        for label, feature in names.items():
            row[f"{label}_auprc"] = float(indexed.loc[feature, "auprc"])
            row[f"{label}_auroc"] = float(indexed.loc[feature, "auroc"])
        row["delta_t_given_cs"] = row["cst_auprc"] - row["cs_auprc"]
        row["delta_t_given_csh"] = row["csht_auprc"] - row["csh_auprc"]
        row["delta_t_given_cs_auroc"] = row["cst_auroc"] - row["cs_auroc"]
        row["delta_t_given_csh_auroc"] = row["csht_auroc"] - row["csh_auroc"]
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)

    paired_rows: list[dict[str, object]] = []
    for key, group in summary.groupby(["task", "context"], sort=True):
        task, context = key
        v1 = group[group["model"] == "v1"].iloc[0]
        random = group[group["model"] == "random"].iloc[0]
        row = {"task": task, "context": context}
        for metric in ("cst_auprc", "csht_auprc", "delta_t_given_cs", "delta_t_given_csh"):
            row[f"v1_{metric}"] = float(v1[metric])
            row[f"random_{metric}"] = float(random[metric])
            row[f"v1_minus_random_{metric}"] = float(v1[metric] - random[metric])
        paired_rows.append(row)
    paired = pd.DataFrame(paired_rows).sort_values(["task", "context"])
    paired.to_csv(args.output.with_name("paired_model_comparison.csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

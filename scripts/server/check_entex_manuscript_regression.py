#!/usr/bin/env python3
"""Check imported manuscript fold metrics; this does not retrain the original probes."""

from pathlib import Path
import argparse
import pandas as pd


def check(root: Path) -> pd.DataFrame:
    rows = []
    expected = {
        "ccre": {"strict": (0.9185, 0.9225), "1hop": (0.9185, 0.9219)},
        "sv": {"strict": (0.8721, 0.9063), "1hop": (0.8721, 0.9122)},
    }
    for task in expected:
        paths = list(root.rglob(f"{task}_fold_metrics.csv"))
        if len(paths) != 1:
            raise ValueError(f"Expected one {task} metrics table: {paths}")
        d = pd.read_csv(paths[0])
        suffix = "_pair" if task == "sv" else ""
        for context, (base, full) in expected[task].items():
            keys = [
                "coordinate_plus_frozen_sequence_fm" + suffix,
                "coordinate_plus_frozen_sequence_fm_plus_frozen_pangenomefm" + suffix,
            ]
            selected = d.loc[d.closure.eq(context) & d.feature_set.isin(keys)]
            if selected.duplicated(["fold", "seed", "feature_set"]).any():
                raise ValueError("Duplicate regression runs")
            wide = selected.pivot(
                index=["fold", "seed"], columns="feature_set", values="auprc"
            )
            if len(wide) != 15 or wide.isna().any().any():
                raise ValueError("Incomplete 5-fold x 3-seed regression")
            observed = wide.mean()
            if (
                abs(observed[keys[0]] - base) > 0.00005
                or abs(observed[keys[1]] - full) > 0.00005
            ):
                raise AssertionError(f"{task}/{context} manuscript regression mismatch")
            rows.append(
                dict(
                    task=task,
                    context=context,
                    n_runs=len(wide),
                    C_S=observed[keys[0]],
                    C_S_T=observed[keys[1]],
                    Delta_T_given_C_S=(wide[keys[1]] - wide[keys[0]]).mean(),
                    source=str(paths[0]),
                    status="pass_cached_result_check",
                )
            )
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-root",
        type=Path,
        default=Path("server_imports/sequence_fm_v2_20260817/extracted"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("results/entex/v1/regression"))
    a = p.parse_args()
    d = check(a.source_root)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    d.to_csv(a.out_dir / "manuscript_regression.csv", index=False)
    print(d.drop(columns="source").to_string(index=False))


if __name__ == "__main__":
    main()

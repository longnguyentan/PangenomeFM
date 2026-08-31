"""Validate and summarize the archived model-capacity comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd


CAPACITY_DEFINITIONS = [
    ("Diagnostic small", "hprc_r2_capacity_tiny_h24_l1", 24, 1, "24 / 1"),
    ("Diagnostic medium", "hprc_r2_capacity_medium_h96_l4", 96, 4, "96 / 4"),
    ("Diagnostic large", "hprc_r2_capacity_large_h192_l6", 192, 6, "192 / 6"),
]
PRINCIPAL_REGIME_ALIASES = (
    "hprc_r2_capacity_principal_h48_l2",
    "hprc_r2",
)


def heldout_rows(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[
        frame["metric_scope"].eq("split")
        & frame["split"].eq("heldout_chr_test")
    ].copy()
    if "closure" in selected:
        selected = selected.loc[selected["closure"].eq("strict")]
    if "experiment_type" in selected:
        selected = selected.loc[selected["experiment_type"].eq("rotating_folds")]
    return selected


def principal_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    selected = heldout_rows(frame)
    for regime in PRINCIPAL_REGIME_ALIASES:
        rows = selected.loc[selected["regime"].eq(regime)].copy()
        if not rows.empty:
            return rows, regime
    return selected.iloc[0:0].copy(), ""


def validate_capacity_evidence(
    old_runs: pd.DataFrame,
    principal_metrics: pd.DataFrame,
) -> dict[str, object]:
    """Require identical held-out fold/seed cells across all configurations."""

    old = heldout_rows(old_runs)
    principal, principal_regime = principal_rows(principal_metrics)
    if principal.empty:
        raise ValueError(
            "principal 48/2 metrics were not found under either archived regime name"
        )
    if principal["auprc"].isna().any():
        raise ValueError("principal 48/2 metrics contain missing AUPRC values")

    key_columns = [
        column for column in ("fold", "seed", "closure") if column in principal
    ]
    if key_columns and principal.duplicated(key_columns).any():
        raise ValueError("principal 48/2 metrics duplicate fold/seed/context cells")
    principal_keys = (
        set(map(tuple, principal[key_columns].itertuples(index=False, name=None)))
        if key_columns
        else set()
    )

    diagnostic_counts: dict[str, int] = {}
    target_totals: dict[str, int] = {}
    for _, regime, _, _, _ in CAPACITY_DEFINITIONS:
        rows = old.loc[old["regime"].eq(regime)].copy()
        if rows.empty:
            raise ValueError(f"capacity metrics are missing regime {regime}")
        if rows["auprc"].isna().any():
            raise ValueError(f"capacity metrics contain missing AUPRC for {regime}")
        diagnostic_counts[regime] = int(len(rows))
        if key_columns:
            if rows.duplicated(key_columns).any():
                raise ValueError(f"capacity metrics duplicate cells for {regime}")
            keys = set(map(tuple, rows[key_columns].itertuples(index=False, name=None)))
            if keys != principal_keys:
                raise ValueError(
                    f"held-out fold/seed/context cells differ between {regime} "
                    f"and principal regime {principal_regime}"
                )
        if key_columns and "n_targets" in rows and "n_targets" in principal:
            comparison = rows[key_columns + ["n_targets"]].merge(
                principal[key_columns + ["n_targets"]],
                on=key_columns,
                suffixes=("_diagnostic", "_principal"),
                validate="one_to_one",
            )
            if not np.array_equal(
                comparison["n_targets_diagnostic"].to_numpy(np.int64),
                comparison["n_targets_principal"].to_numpy(np.int64),
            ):
                raise ValueError(
                    f"held-out target counts differ between {regime} and principal"
                )
            target_totals[regime] = int(rows["n_targets"].sum())
        if key_columns and "positive_fraction" in rows and "positive_fraction" in principal:
            comparison = rows[key_columns + ["positive_fraction"]].merge(
                principal[key_columns + ["positive_fraction"]],
                on=key_columns,
                suffixes=("_diagnostic", "_principal"),
                validate="one_to_one",
            )
            if not np.allclose(
                comparison["positive_fraction_diagnostic"].to_numpy(float),
                comparison["positive_fraction_principal"].to_numpy(float),
                rtol=0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"held-out target prevalence differs between {regime} and principal"
                )

    return {
        "status": "pass",
        "principal_source_regime": principal_regime,
        "principal_runs": int(len(principal)),
        "principal_target_total": (
            int(principal["n_targets"].sum()) if "n_targets" in principal else None
        ),
        "principal_mean_auprc": float(principal["auprc"].mean()),
        "principal_sd_auprc": float(principal["auprc"].std(ddof=1)),
        "diagnostic_run_counts": diagnostic_counts,
        "diagnostic_target_totals": target_totals,
        "comparison_contract": (
            "identical strict-context held-out chromosome fold/seed cells and "
            "target counts"
        ),
    }


def capacity_summary_rows(
    old_runs: pd.DataFrame,
    principal_metrics: pd.DataFrame,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    audit = validate_capacity_evidence(old_runs, principal_metrics)
    old = heldout_rows(old_runs)
    rows: list[dict[str, object]] = []
    for name, regime, hidden, layers, label in CAPACITY_DEFINITIONS:
        values = old.loc[old["regime"].eq(regime), "auprc"].to_numpy(float)
        rows.append(
            {
                "name": name,
                "regime": regime,
                "hidden": hidden,
                "layers": layers,
                "label": label,
                "runs": int(len(values)),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "principal": False,
            }
        )
    principal, principal_regime = principal_rows(principal_metrics)
    values = principal["auprc"].to_numpy(float)
    rows.append(
        {
            "name": "Principal",
            "regime": principal_regime,
            "hidden": 48,
            "layers": 2,
            "label": "48 / 2\n(principal)",
            "runs": int(len(values)),
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "principal": True,
        }
    )
    return rows, audit

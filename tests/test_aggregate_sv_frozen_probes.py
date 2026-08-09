from pathlib import Path

import pandas as pd

from scripts.server.aggregate_sv_frozen_probes import main


def test_sv_aggregator_writes_overall_and_stratum_tables(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "probe"
    for seed, value in [(42, 0.7), (314159, 0.8)]:
        run = root / "fold_a" / f"seed_{seed}" / "strict"
        run.mkdir(parents=True)
        pd.DataFrame(
            {
                "fold": ["fold_a"],
                "seed": [seed],
                "closure": ["strict"],
                "feature_set": ["frozen_pangenomefm"],
                "auroc": [value],
                "auprc": [value],
                "accuracy": [value],
                "precision": [value],
                "recall": [value],
                "f1": [value],
                "nll": [1 - value],
                "brier": [1 - value],
                "ece": [1 - value],
            }
        ).to_csv(run / "metrics.csv", index=False)
        pd.DataFrame(
            {
                "fold": ["fold_a"],
                "seed": [seed],
                "closure": ["strict"],
                "feature_set": ["frozen_pangenomefm"],
                "stratum": ["svtype"],
                "stratum_value": ["INS"],
                "n": [10],
                "auroc": [value],
                "auprc": [value],
                "accuracy": [value],
                "f1": [value],
            }
        ).to_csv(run / "stratified_metrics.csv", index=False)

    out = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "aggregate_sv_frozen_probes.py",
            "--probe-root",
            str(root),
            "--out-dir",
            str(out),
            "--n-bootstrap",
            "20",
            "--seed",
            "7",
        ],
    )
    assert main() == 0
    summary = pd.read_csv(out / "sv_summary.csv")
    strata = pd.read_csv(out / "sv_stratified_summary.csv")
    assert set(summary["metric"]) >= {"auroc", "auprc", "ece"}
    assert strata.loc[0, "n"] == 20
    assert strata.loc[0, "seeds"] == 2

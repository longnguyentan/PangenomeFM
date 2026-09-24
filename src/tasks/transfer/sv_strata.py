"""Analyze frozen SV predictions by native complexity, length, AF and class."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from evaluation.paired_inference import bh_adjust, fold_sign_flip
from scripts.server.run_ccre_frozen_probe_fold import binary_metrics
from scripts.server.run_ccre_frozen_probe_matrix import build_jobs
from tasks.entex.analyze import BASE, FULL, estimate, complexity_labels
from tasks.entex.prepare import fingerprint
from tasks.entex.probe import FEATURES


def annotate(examples: pd.DataFrame, complexity: Path) -> pd.DataFrame:
    if examples.example_id.duplicated().any():
        raise ValueError("Duplicate SV example IDs")
    if (
        not examples.binary_svtype_label.eq(examples.svtype.eq("INS").astype(int)).all()
        or not examples.svtype.isin(["INS", "DEL"]).all()
    ):
        raise ValueError("Expected exact original INS/DEL target")
    x = examples.copy()
    ranks = {"low": 0, "medium": 1, "high": 2}
    for endpoint in ["start", "end"]:
        loci = pd.DataFrame(
            dict(
                locus_id=x.example_id.astype(str),
                chrom=x.chrom,
                start=x[endpoint + "0"],
            )
        )
        x[endpoint + "_complexity"] = complexity_labels(
            loci, complexity
        ).complexity.to_numpy()
    score = (
        x[["start_complexity", "end_complexity"]]
        .apply(lambda values: values.map(ranks))
    )
    x["complexity"] = score.max(axis=1).map({v: k for k, v in ranks.items()})
    x.loc[score.isna().any(axis=1), "complexity"] = "unassigned"
    x["carrier_frequency"] = np.where(
        x.alt_allele_count.eq(1), "singleton_AC1", "AC_gt1"
    )
    x.loc[x.alt_allele_count.isna(), "carrier_frequency"] = "missing"
    return x


def run_metrics(
    predictions: pd.DataFrame, metadata: pd.DataFrame, config: dict
) -> pd.DataFrame:
    x = predictions.merge(metadata, on="example_id", how="left", validate="many_to_one")
    if x.svtype.isna().any() or not x.y_true.eq(x.binary_svtype_label).all():
        raise ValueError("Prediction/metadata mismatch")
    rows = []
    strata = {
        "complexity": "complexity",
        "start_complexity": "start_complexity",
        "length": "length_bin",
        "allele_frequency": "af_bin",
        "carrier_frequency": "carrier_frequency",
        "sv_type": "svtype",
    }
    for (fold, seed, context, feature), values in x.groupby(
        ["fold", "seed", "closure", "feature_set"]
    ):
        for family, column in strata.items():
            for stratum, group in values.groupby(column, dropna=False):
                y = group.y_true.to_numpy(int)
                threshold = float(group.threshold.iloc[0])
                if group.threshold.nunique() != 1:
                    raise ValueError("Threshold changes within a fitted probe")
                result = binary_metrics(y, group.p_calibrated.to_numpy(), threshold)
                prevalence = float(y.mean())
                npositive = int(y.sum())
                eligible = (
                    len(y) >= config["minimum_events_per_run"]
                    and min(npositive, len(y) - npositive)
                    >= config["minimum_per_class_per_run"]
                )
                rows.append(
                    dict(
                        fold=fold,
                        seed=seed,
                        context=context,
                        feature_set=feature,
                        family=family,
                        stratum=str(stratum),
                        positive_count=npositive,
                        negative_count=len(y) - npositive,
                        positive_prevalence=prevalence,
                        sufficiently_powered=eligible,
                        class_recall=float(np.mean(group.y_pred.to_numpy() == y)),
                        balanced_accuracy=float(
                            balanced_accuracy_score(y, group.y_pred)
                        )
                        if len(np.unique(y)) == 2
                        else np.nan,
                        normalized_ap=(result["auprc"] - prevalence) / (1 - prevalence)
                        if 0 < prevalence < 1
                        else np.nan,
                        **result,
                    )
                )
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, interactions = [], []
    for (family, stratum, context), group in metrics.groupby(
        ["family", "stratum", "context"]
    ):
        wide = group.pivot(
            index=["fold", "seed"], columns="feature_set", values="auprc"
        )
        if BASE not in wide or FULL not in wide:
            raise ValueError("Missing paired feature sets")
        gains = (wide[FULL] - wide[BASE]).rename("gain").reset_index()
        base = group.loc[group.feature_set.eq(BASE)]
        row = dict(
            family=family,
            stratum=stratum,
            context=context,
            mean_n=float(base.n.mean()),
            mean_prevalence=float(base.positive_prevalence.mean()),
            minimum_positive=int(base.positive_count.min()),
            minimum_negative=int(base.negative_count.min()),
            sufficiently_powered=bool(base.sufficiently_powered.all()),
            p_fold_signflip=fold_sign_flip(gains),
            **estimate(gains, "gain", config["n_bootstrap"], config["seed"]),
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["q_bh"] = summary.groupby(["family", "context"]).p_fold_signflip.transform(
        bh_adjust
    )
    for context, frame in metrics.loc[metrics.family.eq("complexity")].groupby(
        "context"
    ):
        wide = frame.pivot(
            index=["fold", "seed", "stratum"], columns="feature_set", values="auprc"
        )
        gain = (wide[FULL] - wide[BASE]).unstack("stratum")
        difference = (gain.high - gain.low).rename("gain").reset_index()
        interactions.append(
            dict(
                context=context,
                contrast="high_minus_low_topology_gain",
                p_fold_signflip=fold_sign_flip(difference),
                **estimate(difference, "gain", config["n_bootstrap"], config["seed"]),
            )
        )
    return summary, pd.DataFrame(interactions)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-root", type=Path, required=True)
    ap.add_argument("--examples", type=Path, required=True)
    ap.add_argument("--complexity", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--config", type=Path, default=Path("configs/structural_mechanism_v1.json")
    )
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    paths = sorted(args.probe_root.glob("fold_*/seed_*/*/test_predictions.csv.gz"))
    jobs = build_jobs(
        json.loads(Path("configs/server_full_multicohort_20260806.json").read_text())
    )
    expected = {(j.fold, j.seed, j.closure) for j in jobs}
    metadata = annotate(pd.read_csv(args.examples), args.complexity)
    if metadata.complexity.ne("unassigned").mean() < 0.95:
        raise ValueError("Native complexity assignment below 95%")
    frames, seen = [], set()
    for path in paths:
        predictions = pd.read_csv(path)
        predictions["feature_set"] = predictions.feature_set.str.removesuffix("_pair")
        predictions = predictions.loc[predictions.feature_set.isin(FEATURES)]
        key = tuple(predictions[["fold", "seed", "closure"]].drop_duplicates().iloc[0])
        if (
            key in seen
            or len(predictions[["fold", "seed", "closure"]].drop_duplicates()) != 1
        ):
            raise ValueError("Duplicate or mixed runs")
        seen.add(key)
        if predictions.duplicated(["example_id", "feature_set"]).any():
            raise ValueError("Duplicate predictions")
        if set(predictions.feature_set) != set(FEATURES):
            raise ValueError("Missing requested seven feature sets")
        if (
            not predictions.groupby("example_id")
            .feature_set.nunique()
            .eq(len(FEATURES))
            .all()
        ):
            raise ValueError("Feature universes differ")
        frames.append(run_metrics(predictions, metadata, cfg))
        print(key, "complete", flush=True)
    if seen != expected:
        raise ValueError("Incomplete manuscript matrix")
    metrics = pd.concat(frames, ignore_index=True)
    summary, interaction = summarize(metrics, cfg)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(args.out_dir / "per_run.csv", index=False)
    summary.to_csv(args.out_dir / "paired_gains.csv", index=False)
    interaction.to_csv(args.out_dir / "complexity_interaction.csv", index=False)
    metadata.groupby(
        ["svtype", "complexity", "length_bin", "af_bin"], dropna=False
    ).size().rename("n").to_csv(args.out_dir / "counts.csv")
    (args.out_dir / "audit.json").write_text(
        json.dumps(
            dict(
                status="complete",
                predictions_refitted=False,
                runs=len(seen),
                configuration=cfg,
                examples=fingerprint(args.examples),
                complexity=fingerprint(args.complexity),
                complexity_assigned_fraction=float(
                    metadata.complexity.ne("unassigned").mean()
                ),
                within_class_note="SVTYPE is the target: AP/AUROC undefined; class_recall is sensitivity for INS and specificity for DEL",
                multiplicity_note="BH over exact two-sided fold sign-flip p-values within context/family; exploratory, five-fold resolution and overlapping CV training limit inference",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

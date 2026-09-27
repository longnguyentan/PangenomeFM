"""Evaluate frozen SNV predictions with equal locus weights and fixed donor/tissue strata.

This changes evaluation only. Donor/tissue strata retain chromosome-held-out
predictions; they are not donor/tissue-held-out models.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from evaluation.calibration import EPSILON

from tasks.entex.analyze import BASE, FULL, estimate, paired, validate_comparator_loci
from tasks.entex.prepare import fingerprint
from tasks.entex.snv import ASSAYS


def locus_weights(frame: pd.DataFrame) -> np.ndarray:
    if frame.measurement_id.duplicated().any() or frame.locus_id.isna().any():
        raise ValueError("Repeated measurement or missing locus identity")
    return 1.0 / frame.groupby("locus_id").locus_id.transform("size").to_numpy()


def scores(frame: pd.DataFrame, equal_locus: bool = False) -> dict:
    weights = locus_weights(frame) if equal_locus else np.ones(len(frame))
    prevalence = float(np.average(frame.y_true, weights=weights))
    result = dict(
        n=len(frame),
        n_loci=frame.locus_id.nunique(),
        n_positive=int(frame.y_true.sum()),
        positive_prevalence=prevalence,
        weight_sum=float(weights.sum()),
    )
    if frame.y_true.nunique() != 2:
        return dict(result, auprc=np.nan, auroc=np.nan, normalized_ap=np.nan)
    # Match manuscript binary_metrics, including ties from saturated probabilities.
    probability = np.clip(frame.p_calibrated.to_numpy(float), EPSILON, 1 - EPSILON)
    ap = float(
        average_precision_score(frame.y_true, probability, sample_weight=weights)
    )
    return dict(
        result,
        auprc=ap,
        auroc=float(roc_auc_score(frame.y_true, probability, sample_weight=weights)),
        normalized_ap=(ap - prevalence) / (1 - prevalence),
    )


def identity_digest(frame: pd.DataFrame) -> str:
    columns = ["measurement_id", "locus_id", "chrom", "y_true", "donor", "tissue"]
    if frame[columns].isna().any().any() or frame.measurement_id.duplicated().any():
        raise ValueError("Missing metadata or duplicate measurement")
    content = frame[columns].sort_values("measurement_id").to_csv(index=False)
    return hashlib.sha256(content.encode()).hexdigest()


def eligibility(counts: pd.DataFrame, folds: list[str], protocol: dict) -> pd.DataFrame:
    """Decide using counts from every held-out fold, without reading scores."""
    if counts.duplicated(["stratum", "group", "fold"]).any():
        raise ValueError("Duplicate stratum counts")
    rows = []
    for (stratum, group), values in counts.groupby(["stratum", "group"]):
        values = values.set_index("fold").reindex(folds, fill_value=0)
        minimum_n = int(values.n.min())
        minimum_positive = int(values.n_positive.min())
        minimum_negative = int((values.n - values.n_positive).min())
        rows.append(
            dict(
                stratum=stratum,
                group=group,
                min_fold_n=minimum_n,
                min_fold_positive=minimum_positive,
                min_fold_negative=minimum_negative,
                eligible=(
                    minimum_n >= protocol["minimum_measurements_per_test_fold"]
                    and minimum_positive >= protocol["minimum_positive_per_test_fold"]
                    and minimum_negative >= protocol["minimum_negative_per_test_fold"]
                ),
            )
        )
    return pd.DataFrame(rows)


def run(root: Path, out: Path, protocol_path: Path, assay: str) -> None:
    protocol = json.loads(protocol_path.read_text())["prediction_followups"]
    if assay not in protocol["assays"]:
        raise ValueError("Assay not in follow-up protocol")
    base_config = json.loads(Path("configs/entex_v1.json").read_text())
    manuscript = json.loads(Path(base_config["manuscript_config"]).read_text())
    folds = [f["name"] for f in manuscript["rotating_chromosome_folds"]]
    seeds, contexts = (
        manuscript["training"]["seeds"],
        manuscript["training"]["contexts"],
    )
    paths = sorted(root.glob("fold_*/seed_*/*/predictions.parquet"))
    observed = {
        (p.parents[2].name, int(p.parents[1].name.removeprefix("seed_")), p.parent.name)
        for p in paths
    }
    if observed != set(product(folds, seeds, contexts)):
        raise ValueError("Incomplete or unexpected fold/seed/context prediction matrix")
    out.mkdir(parents=True, exist_ok=False)
    counts, identities, seen_loci = [], {}, set()
    metadata = ["measurement_id", "locus_id", "chrom", "y_true", "donor", "tissue"]
    for fold in manuscript["rotating_chromosome_folds"]:
        p = (
            root
            / fold["name"]
            / f"seed_{seeds[0]}"
            / contexts[0]
            / "predictions.parquet"
        )
        base = pd.read_parquet(
            p, columns=metadata, filters=[("feature_set", "==", BASE)]
        )
        if not set(base.chrom) <= set(fold["test"]):
            raise ValueError("Prediction chromosome is outside held-out fold")
        if seen_loci & set(base.locus_id):
            raise ValueError("A locus occurs in multiple held-out folds")
        seen_loci.update(base.locus_id)
        identities[fold["name"]] = identity_digest(base)
        for stratum in protocol["strata"]:
            for group, values in base.groupby(stratum):
                counts.append(
                    dict(
                        fold=fold["name"],
                        stratum=stratum,
                        group=group,
                        n=len(values),
                        n_positive=int(values.y_true.sum()),
                    )
                )
    counts = pd.DataFrame(counts)
    eligible = eligibility(counts, folds, protocol)
    counts.to_csv(out / "eligibility_counts.csv", index=False)
    eligible.to_csv(out / "eligibility.csv", index=False)
    rows, provenance = [], []
    for path in paths:
        fold, seed, context = (
            path.parents[2].name,
            int(path.parents[1].name.removeprefix("seed_")),
            path.parent.name,
        )
        predictions = pd.read_parquet(
            path,
            columns=metadata + ["feature_set", "p_calibrated"],
            filters=[("feature_set", "in", [BASE, FULL])],
        )
        validate_comparator_loci(predictions)
        original_metrics = pd.read_csv(path.parent / "metrics.csv").set_index(
            "feature_set"
        )
        for feature, frame in predictions.groupby("feature_set"):
            unweighted = scores(frame)
            for metric in ["auprc", "auroc", "positive_prevalence"]:
                if not np.isclose(
                    unweighted[metric],
                    original_metrics.loc[feature, metric],
                    atol=1e-12,
                    rtol=0,
                ):
                    raise ValueError(
                        f"Measurement-weight control does not reproduce original {metric}: {path}"
                    )
            if identity_digest(frame) != identities[fold]:
                raise ValueError(
                    "Labels/identities/metadata changed across seeds, contexts or comparators"
                )
            common = dict(
                assay=assay, fold=fold, seed=seed, closure=context, feature_set=feature
            )
            for analysis in ["measurement_weight", "equal_locus_weight"]:
                rows.append(
                    dict(
                        common,
                        analysis=analysis,
                        group="all",
                        **scores(frame, equal_locus=analysis == "equal_locus_weight"),
                    )
                )
            for row in eligible.loc[eligible.eligible].itertuples():
                values = frame.loc[frame[row.stratum].eq(row.group)]
                rows.append(
                    dict(
                        common, analysis=row.stratum, group=row.group, **scores(values)
                    )
                )
        provenance.append(
            dict(
                path=str(path),
                bytes=path.stat().st_size,
                mtime_ns=path.stat().st_mtime_ns,
                fold=fold,
                seed=seed,
                context=context,
                targets_sha256=identities[fold],
            )
        )
    runs = pd.DataFrame(rows)
    metrics = ["auprc", "auroc", "normalized_ap", "positive_prevalence"]
    for stratum in protocol["strata"]:
        selected = runs.loc[runs.analysis.eq(stratum)]
        if selected.empty:
            continue
        macro = selected.groupby(
            ["assay", "fold", "seed", "closure", "feature_set"], as_index=False
        )[metrics].mean()
        macro["analysis"], macro["group"] = f"{stratum}_macro", "eligible_equal_weight"
        macro["n_groups"] = int(
            eligible.loc[eligible.stratum.eq(stratum) & eligible.eligible].shape[0]
        )
        runs = pd.concat([runs, macro], ignore_index=True)
    absolute, gains = [], []
    for (analysis, group, context), values in runs.groupby(
        ["analysis", "group", "closure"]
    ):
        common = dict(assay=assay, analysis=analysis, group=group, context=context)
        for metric in metrics:
            for feature, feature_values in values.groupby("feature_set"):
                absolute.append(
                    dict(
                        common,
                        feature_set=feature,
                        metric=metric,
                        **estimate(
                            feature_values,
                            metric,
                            protocol["n_bootstrap"],
                            protocol["bootstrap_seed"],
                        ),
                    )
                )
            if metric != "positive_prevalence":
                gains.append(
                    dict(
                        common,
                        metric=metric,
                        comparison="Delta_T_given_C_S",
                        **paired(
                            values,
                            BASE,
                            protocol["n_bootstrap"],
                            protocol["bootstrap_seed"],
                            metric=metric,
                        ),
                    )
                )
    runs.to_csv(out / "per_run.csv", index=False)
    pd.DataFrame(absolute).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(gains).to_csv(out / "paired_gains.csv", index=False)
    pd.DataFrame(provenance).to_csv(out / "prediction_audit.csv", index=False)
    (out / "audit.json").write_text(
        json.dumps(
            dict(
                status="complete",
                assay=assay,
                protocol=fingerprint(protocol_path),
                n_prediction_runs=len(paths),
                n_loci=len(seen_loci),
                encoders_refitted=False,
                probes_refitted=False,
                eligible_groups=eligible.to_dict("records"),
                limitations=protocol["limitations"],
            ),
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--assay", choices=list(ASSAYS), required=True)
    ap.add_argument(
        "--protocol", type=Path, default=Path("configs/entex_meeting_20260929.json")
    )
    args = ap.parse_args()
    run(args.probe_root, args.out_dir, args.protocol, args.assay)


if __name__ == "__main__":
    main()

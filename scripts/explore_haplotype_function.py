from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
csv.field_size_limit(sys.maxsize)


TEST_CHROMS = {"chr8", "chr19", "chr22"}
VAL_CHROMS = {"chr16"}


REF_PREFIXES = ("GRCh38", "CHM13", "id=CHM13")


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return path.open("r", newline="")


def _sample_id(sn: str) -> str:
    if "#" in sn:
        return sn.split("#", 1)[0]
    if "|" in sn:
        return sn.split("|", 1)[0]
    if "." in sn:
        return sn.split(".", 1)[0]
    return sn


def _chrom_from_sn(sn: str) -> str:
    if "#" in sn:
        return sn.split("#")[-1]
    if "|" in sn:
        return sn.split("|")[-1]
    return sn


def _is_reference(sn: str, sr: int) -> bool:
    return sr == 0 or sn.startswith(REF_PREFIXES)


def read_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="infer")
    df["segid"] = df["segid"].astype(int)
    df["is_ccre"] = (df["ccre_class"] != "background").astype(int)
    df["is_enhancer_like"] = df["ccre_class"].isin(["dELS", "pELS"]).astype(int)
    df["is_dels"] = (df["ccre_class"] == "dELS").astype(int)
    df["is_pels"] = (df["ccre_class"] == "pELS").astype(int)
    df["is_promoter_like"] = df["ccre_class"].isin(["PLS", "CA-H3K4me3"]).astype(int)
    df["is_tf_ctcf"] = df["ccre_class"].isin(["TF", "CA-TF", "CA-CTCF"]).astype(int)
    df["is_open_chromatin"] = (df["ccre_class"] == "CA").astype(int)
    return df


def stream_segments(
    segments_path: Path, labeled_segids: set[int]
) -> tuple[dict[str, int], dict[str, tuple[int, str, int, int, int, bool]], dict[int, str]]:
    name_to_idx: dict[str, int] = {}
    name_to_props: dict[str, tuple[int, str, int, int, int, bool]] = {}
    labeled_idx_to_name: dict[int, str] = {}

    with segments_path.open("r", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        required = {"name", "LN", "SN", "SO", "SR"}
        missing = required - set(col)
        if missing:
            raise ValueError(f"{segments_path} missing columns: {sorted(missing)}")

        for row_idx, row in enumerate(reader):
            name = row[col["name"]]
            sn = row[col["SN"]]
            so = int(row[col["SO"]] or 0)
            ln = int(row[col["LN"]] or 0)
            sr = int(row[col["SR"]] or 0)
            is_ref = _is_reference(sn, sr)
            name_to_idx[name] = row_idx
            name_to_props[name] = (row_idx, sn, so, ln, sr, is_ref)
            if row_idx in labeled_segids:
                labeled_idx_to_name[row_idx] = name

    missing_labeled = labeled_segids - set(labeled_idx_to_name)
    if missing_labeled:
        raise ValueError(f"Could not resolve {len(missing_labeled)} labeled segment ids")
    return name_to_idx, name_to_props, labeled_idx_to_name


def build_haplotype_features(
    links_path: Path,
    labels: pd.DataFrame,
    name_to_props: dict[str, tuple[int, str, int, int, int, bool]],
    labeled_idx_to_name: dict[int, str],
) -> pd.DataFrame:
    labeled_name_to_idx = {name: idx for idx, name in labeled_idx_to_name.items()}
    accum: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "degree_total": 0,
            "ref_neighbor_count": 0,
            "alt_neighbor_count": 0,
            "alt_edge_count": 0,
            "alt_neighbor_len_sum": 0,
            "alt_neighbor_len_max": 0,
            "alt_same_chrom_count": 0,
            "alt_min_abs_delta_so": math.nan,
            "alt_sn_set": set(),
            "alt_sample_set": set(),
        }
    )

    def update(labeled_name: str, other_name: str, edge_sr: int) -> None:
        segid = labeled_name_to_idx[labeled_name]
        if other_name not in name_to_props:
            return
        _, other_sn, other_so, other_ln, other_sr, other_is_ref = name_to_props[other_name]
        d = accum[segid]
        d["degree_total"] += 1
        if other_is_ref:
            d["ref_neighbor_count"] += 1
            return

        d["alt_neighbor_count"] += 1
        d["alt_edge_count"] += int(edge_sr != 0)
        d["alt_neighbor_len_sum"] += other_ln
        d["alt_neighbor_len_max"] = max(d["alt_neighbor_len_max"], other_ln)
        d["alt_sn_set"].add(other_sn)
        d["alt_sample_set"].add(_sample_id(other_sn))
        label_row = label_meta[segid]
        if _chrom_from_sn(other_sn) == label_row["chrom"]:
            d["alt_same_chrom_count"] += 1
            delta = abs(int(label_row["SO"]) - other_so)
            old = d["alt_min_abs_delta_so"]
            d["alt_min_abs_delta_so"] = delta if math.isnan(old) else min(old, delta)

    label_meta = labels.set_index("segid")[["chrom", "SO"]].to_dict("index")
    with links_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            a = row["from_seg"]
            b = row["to_seg"]
            edge_sr = int(row.get("SR") or 0)
            if a in labeled_name_to_idx:
                update(a, b, edge_sr)
            if b in labeled_name_to_idx:
                update(b, a, edge_sr)

    rows: list[dict[str, Any]] = []
    for segid in labels["segid"].astype(int):
        d = accum[segid]
        alt_count = int(d["alt_neighbor_count"])
        degree = int(d["degree_total"])
        rows.append(
            {
                "segid": segid,
                "degree_total": degree,
                "ref_neighbor_count": int(d["ref_neighbor_count"]),
                "alt_neighbor_count": alt_count,
                "has_alt_neighbor": int(alt_count > 0),
                "alt_neighbor_fraction": alt_count / degree if degree else 0.0,
                "alt_edge_count": int(d["alt_edge_count"]),
                "alt_neighbor_len_sum": int(d["alt_neighbor_len_sum"]),
                "alt_neighbor_len_max": int(d["alt_neighbor_len_max"]),
                "alt_unique_sn_count": len(d["alt_sn_set"]),
                "alt_unique_sample_count": len(d["alt_sample_set"]),
                "alt_same_chrom_count": int(d["alt_same_chrom_count"]),
                "alt_min_abs_delta_so": d["alt_min_abs_delta_so"],
            }
        )
    out = pd.DataFrame(rows)
    out["alt_min_abs_delta_so"] = out["alt_min_abs_delta_so"].fillna(-1)
    return out


def _metric_dict(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    out = {
        "n": int(len(y_true)),
        "positive_fraction": float(np.mean(y_true)),
        "auroc": float(roc_auc_score(y_true, prob)) if len(set(y_true)) == 2 else float("nan"),
        "auprc": float(average_precision_score(y_true, prob)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
    }
    return out


def fit_eval(
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    categorical_cols: list[str],
    seed: int,
) -> dict[str, Any]:
    train = df[~df["chrom"].isin(TEST_CHROMS | VAL_CHROMS)].copy()
    test = df[df["chrom"].isin(TEST_CHROMS)].copy()
    if train[target].nunique() < 2 or test[target].nunique() < 2:
        return {"target": target, "skipped": True}

    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    clf = Pipeline(
        [
            ("pre", pre),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )
    clf.fit(train[feature_cols], train[target])
    prob = clf.predict_proba(test[feature_cols])[:, 1]
    return {
        "target": target,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "features": feature_cols,
        "metrics": _metric_dict(test[target].to_numpy(), prob),
    }


def shuffle_hap_features_within_chrom(df: pd.DataFrame, hap_cols: list[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    for chrom, idx in out.groupby("chrom").groups.items():
        idx_list = list(idx)
        if len(idx_list) < 2:
            continue
        perm = rng.permutation(idx_list)
        out.loc[idx_list, hap_cols] = out.loc[perm, hap_cols].to_numpy()
    return out


def association_tests(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in targets:
        tab = pd.crosstab(df["has_alt_neighbor"], df[target])
        for value in [0, 1]:
            if value not in tab.index:
                tab.loc[value] = [0] * len(tab.columns)
        for value in [0, 1]:
            if value not in tab.columns:
                tab[value] = 0
        tab = tab.sort_index().sort_index(axis=1)
        oddsratio, pvalue = fisher_exact(
            [
                [int(tab.loc[1, 1]), int(tab.loc[1, 0])],
                [int(tab.loc[0, 1]), int(tab.loc[0, 0])],
            ]
        )
        rows.append(
            {
                "target": target,
                "n_alt_context": int(tab.loc[1].sum()),
                "n_no_alt_context": int(tab.loc[0].sum()),
                "positive_rate_alt_context": float(tab.loc[1, 1] / max(tab.loc[1].sum(), 1)),
                "positive_rate_no_alt_context": float(tab.loc[0, 1] / max(tab.loc[0].sum(), 1)),
                "odds_ratio": float(oddsratio),
                "fisher_pvalue": float(pvalue),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(
    out_path: Path,
    assoc: pd.DataFrame,
    pred: pd.DataFrame,
    shuffle: pd.DataFrame,
) -> None:
    def table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in df.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    vals.append(f"{val:.4g}")
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = [
        "# Haplotype-Context Functional Prediction Exploration",
        "",
        "This exploratory analysis tests whether graph neighborhoods that include",
        "alternate-haplotype context carry functional signal for ENCODE cCRE labels",
        "mapped to HPRC GRCh38-path nodes.",
        "",
        "Important scope note: these are not haplotype-specific functional labels.",
        "The labels are reference-projected cCRE annotations; the haplotype signal",
        "comes from adjacent non-reference graph context around each labeled node.",
        "",
        "## Association With Alternate-Haplotype Context",
        "",
        table(assoc),
        "",
        "## Functional Prediction",
        "",
        table(pred),
        "",
        "## Within-Chromosome Haplotype-Feature Shuffle Control",
        "",
        table(shuffle),
        "",
        "## Interpretation",
        "",
        "- If haplotype-context features improve over reference-local features and",
        "  outperform the within-chromosome shuffle, they provide evidence that",
        "  alternate-haplotype graph neighborhoods carry functional information.",
        "- This should be described as haplotype-context-aware functional prediction,",
        "  not haplotype-specific functional prediction, until path-resolved assay",
        "  labels or phased functional labels are available.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--segments", default="data/hprc/full_segments.csv")
    ap.add_argument("--links", default="data/hprc/full_links.csv")
    ap.add_argument("--labels", default="data/hprc/ccre/run_003/node_labels.csv.gz")
    ap.add_argument("--out-dir", default="results/hprc/haplotype_function_exploration")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = read_labels(ROOT / args.labels)
    labeled_segids = set(labels["segid"].astype(int))
    _, name_to_props, labeled_idx_to_name = stream_segments(ROOT / args.segments, labeled_segids)
    hap = build_haplotype_features(ROOT / args.links, labels, name_to_props, labeled_idx_to_name)

    df = labels.merge(hap, on="segid", how="left")
    df["log1p_SO"] = np.log1p(df["SO"].clip(lower=0))
    df["log1p_LN"] = np.log1p(df["LN"].clip(lower=0))
    df["log1p_degree_total"] = np.log1p(df["degree_total"])
    df["log1p_alt_neighbor_count"] = np.log1p(df["alt_neighbor_count"])
    df["log1p_alt_neighbor_len_sum"] = np.log1p(df["alt_neighbor_len_sum"])
    df["log1p_alt_neighbor_len_max"] = np.log1p(df["alt_neighbor_len_max"])
    df["log1p_alt_unique_sample_count"] = np.log1p(df["alt_unique_sample_count"])
    valid_delta = df["alt_min_abs_delta_so"].clip(lower=0)
    df["log1p_alt_min_abs_delta_so"] = np.where(df["alt_min_abs_delta_so"] < 0, 0.0, np.log1p(valid_delta))

    targets = [
        "is_ccre",
        "is_enhancer_like",
        "is_dels",
        "is_pels",
        "is_promoter_like",
        "is_tf_ctcf",
        "is_open_chromatin",
    ]
    assoc = association_tests(df, targets)

    ref_features = ["log1p_SO", "log1p_LN", "chrom"]
    local_features = ref_features + ["log1p_degree_total"]
    hap_features = [
        "has_alt_neighbor",
        "alt_neighbor_fraction",
        "log1p_alt_neighbor_count",
        "log1p_alt_neighbor_len_sum",
        "log1p_alt_neighbor_len_max",
        "log1p_alt_unique_sample_count",
        "alt_same_chrom_count",
        "log1p_alt_min_abs_delta_so",
    ]
    feature_sets = {
        "reference_local": ref_features,
        "reference_local_plus_degree": local_features,
        "haplotype_context_only": hap_features,
        "reference_plus_haplotype_context": ref_features + hap_features,
        "reference_degree_plus_haplotype_context": local_features + hap_features,
    }

    pred_rows: list[dict[str, Any]] = []
    for target in targets:
        for feature_set, cols in feature_sets.items():
            res = fit_eval(df, target, cols, categorical_cols=["chrom"] if "chrom" in cols else [], seed=args.seed)
            if res.get("skipped"):
                continue
            pred_rows.append(
                {
                    "target": target,
                    "feature_set": feature_set,
                    "n_train": res["n_train"],
                    "n_test": res["n_test"],
                    **res["metrics"],
                }
            )
    pred = pd.DataFrame(pred_rows)

    shuffled = shuffle_hap_features_within_chrom(df, hap_features, args.seed)
    shuffle_rows: list[dict[str, Any]] = []
    for target in targets:
        base = fit_eval(
            df,
            target,
            ref_features + hap_features,
            categorical_cols=["chrom"],
            seed=args.seed,
        )
        shuf = fit_eval(
            shuffled,
            target,
            ref_features + hap_features,
            categorical_cols=["chrom"],
            seed=args.seed,
        )
        if base.get("skipped") or shuf.get("skipped"):
            continue
        shuffle_rows.append(
            {
                "target": target,
                "real_auroc": base["metrics"]["auroc"],
                "shuffled_auroc": shuf["metrics"]["auroc"],
                "delta_auroc": base["metrics"]["auroc"] - shuf["metrics"]["auroc"],
                "real_auprc": base["metrics"]["auprc"],
                "shuffled_auprc": shuf["metrics"]["auprc"],
                "delta_auprc": base["metrics"]["auprc"] - shuf["metrics"]["auprc"],
            }
        )
    shuffle = pd.DataFrame(shuffle_rows)

    df.to_csv(out_dir / "node_haplotype_context_features.csv.gz", index=False, compression="gzip")
    assoc.to_csv(out_dir / "association_by_alt_context.csv", index=False)
    pred.to_csv(out_dir / "prediction_feature_set_comparison.csv", index=False)
    shuffle.to_csv(out_dir / "within_chrom_hap_context_shuffle.csv", index=False)
    payload = {
        "inputs": vars(args),
        "n_labeled_nodes": int(len(df)),
        "n_with_alt_neighbor": int(df["has_alt_neighbor"].sum()),
        "n_without_alt_neighbor": int((1 - df["has_alt_neighbor"]).sum()),
        "test_chroms": sorted(TEST_CHROMS),
        "val_chroms": sorted(VAL_CHROMS),
        "feature_sets": feature_sets,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(out_dir / "summary.md", assoc, pred, shuffle)

    print(f"wrote {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

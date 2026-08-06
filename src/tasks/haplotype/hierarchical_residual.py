"""Held-out-donor/population training for hierarchical graph residual prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from models.hierarchical_path import HierarchicalPathEncoder


def _embedding_lookup(path: str | Path) -> dict[str, np.ndarray]:
    payload = np.load(path)
    return {
        str(key): value.astype(np.float32)
        for key, value in zip(payload["ids"], payload["embeddings"])
    }


def _paired_embedding_deltas(
    pairs: pd.DataFrame, path: str | Path
) -> np.ndarray:
    lookup = _embedding_lookup(path)
    missing = sorted(
        (
            set(pairs["h1_embedding_id"].astype(str))
            | set(pairs["h2_embedding_id"].astype(str))
        )
        - set(lookup)
    )
    if missing:
        raise ValueError(f"{path} lacks {len(missing)} pair embeddings.")
    return np.stack(
        [
            lookup[str(left)] - lookup[str(right)]
            for left, right in zip(
                pairs["h1_embedding_id"], pairs["h2_embedding_id"]
            )
        ]
    )


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rho = (
        spearmanr(y, pred).statistic
        if len(y) > 1 and np.ptp(y) > 0 and np.ptp(pred) > 0
        else float("nan")
    )
    return {
        "spearman": float(rho),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else float("nan"),
        "direction_accuracy": float(np.mean((y > 0) == (pred > 0))),
    }


def _paired_group_bootstrap(
    per_group: pd.DataFrame,
    *,
    baseline: str,
    candidate: str,
    seed: int,
    n_bootstrap: int = 10_000,
) -> list[dict[str, Any]]:
    """Bootstrap held-out groups, preserving paired model comparisons."""

    results: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for metric in ("spearman", "mae", "r2", "direction_accuracy"):
        pivot = per_group.pivot(
            index="heldout_group", columns="model", values=metric
        )
        pivot = pivot[[baseline, candidate]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if metric == "mae":
            differences = (
                pivot[baseline].to_numpy() - pivot[candidate].to_numpy()
            )
            orientation = "baseline_minus_candidate; positive favors candidate"
        else:
            differences = (
                pivot[candidate].to_numpy() - pivot[baseline].to_numpy()
            )
            orientation = "candidate_minus_baseline; positive favors candidate"
        if len(differences):
            draws = rng.choice(
                differences,
                size=(n_bootstrap, len(differences)),
                replace=True,
            ).mean(axis=1)
            lower, upper = np.quantile(draws, [0.025, 0.975])
            observed = float(differences.mean())
        else:
            observed = lower = upper = float("nan")
        results.append(
            {
                "metric": metric,
                "n_groups": int(len(differences)),
                "difference": observed,
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
                "orientation": orientation,
                "bootstrap_replicates": n_bootstrap,
            }
        )
    return results


def _select_residual_shrinkage(
    y_true: np.ndarray,
    baseline_prediction: np.ndarray,
    residual_prediction: np.ndarray,
    *,
    grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> tuple[float, list[dict[str, float]]]:
    """Select residual strength using validation MAE only.

    Ties favor the smaller residual weight. This keeps calibration repair
    independent of the held-out donor or population and makes alpha=0 an
    explicit valid outcome when the residual does not generalize.
    """

    sweep: list[dict[str, float]] = []
    for alpha in grid:
        prediction = baseline_prediction + alpha * residual_prediction
        sweep.append(
            {
                "alpha": float(alpha),
                "mae": float(mean_absolute_error(y_true, prediction)),
                "r2": float(r2_score(y_true, prediction))
                if len(y_true) > 1
                else float("nan"),
            }
        )
    selected = min(sweep, key=lambda row: (row["mae"], row["alpha"]))
    return float(selected["alpha"]), sweep


def _fit_predict_antisymmetric_histgb(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *prediction_features: np.ndarray,
    seed: int,
) -> list[np.ndarray]:
    """Fit a nonlinear sequence baseline with exact swap antisymmetry."""

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(
        np.concatenate([train_x, -train_x], axis=0),
        np.concatenate([train_y, -train_y], axis=0),
    )
    return [
        0.5 * (model.predict(values) - model.predict(-values))
        for values in prediction_features
    ]


def _group_tensors(
    frame: pd.DataFrame, device: str, event_bp: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    donor_codes, donors = pd.factorize(frame["donor_id"].astype(str), sort=True)
    event_keys = (
        frame["donor_id"].astype(str)
        + ":"
        + frame["h1_contig"].astype(str)
        + ":"
        + (frame["h1_window_start"].astype(int) // event_bp).astype(str)
    )
    event_codes, event_uniques = pd.factorize(event_keys, sort=True)
    event_to_donor: list[int] = []
    donor_by_name = {name: idx for idx, name in enumerate(donors)}
    for event in event_uniques:
        event_to_donor.append(donor_by_name[str(event).split(":", 1)[0]])
    return (
        torch.arange(len(frame), dtype=torch.long, device=device),
        torch.tensor(event_codes, dtype=torch.long, device=device),
        torch.tensor(event_to_donor, dtype=torch.long, device=device),
    )


class HierarchicalResidualNetwork(torch.nn.Module):
    """Window -> event -> path residual with exact H1/H2 antisymmetry."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        swap_signs: np.ndarray | None = None,
    ):
        super().__init__()
        self.encoder = HierarchicalPathEncoder(input_dim, hidden_dim, output_dim)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + 2 * output_dim, hidden_dim, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1, bias=False),
        )
        if swap_signs is None:
            swap_signs = -np.ones(input_dim, dtype=np.float32)
        swap_signs = np.asarray(swap_signs, dtype=np.float32)
        if swap_signs.shape != (input_dim,) or not np.isin(
            swap_signs, (-1.0, 1.0)
        ).all():
            raise ValueError("swap_signs must contain one +/-1 value per feature.")
        self.register_buffer(
            "swap_signs",
            torch.tensor(swap_signs, dtype=torch.float32),
        )

    def _raw(
        self,
        values: torch.Tensor,
        node_to_window: torch.Tensor,
        window_to_event: torch.Tensor,
        event_to_path: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.encoder(
            values, node_to_window, window_to_event, event_to_path
        )
        event_for_window = encoded["event_embeddings"][window_to_event]
        path_for_window = encoded["path_embeddings"][
            event_to_path[window_to_event]
        ]
        combined = torch.cat(
            [encoded["window_embeddings"], event_for_window, path_for_window],
            dim=1,
        )
        return self.head(combined).squeeze(1)

    def forward(
        self,
        values: torch.Tensor,
        node_to_window: torch.Tensor,
        window_to_event: torch.Tensor,
        event_to_path: torch.Tensor,
    ) -> torch.Tensor:
        swapped = values * self.swap_signs
        return 0.5 * (
            self._raw(
                values,
                node_to_window,
                window_to_event,
                event_to_path,
            )
            - self._raw(
                swapped,
                node_to_window,
                window_to_event,
                event_to_path,
            )
        )


def _fit_residual(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    train_x: np.ndarray,
    val_x: np.ndarray,
    train_y: np.ndarray,
    val_y: np.ndarray,
    *,
    device: str,
    event_bp: int,
    seed: int,
    epochs: int,
    patience: int,
    swap_signs: np.ndarray | None = None,
) -> tuple[HierarchicalResidualNetwork, np.ndarray]:
    torch.manual_seed(seed)
    scale = np.sqrt(np.mean(np.square(train_x), axis=0))
    scale[scale < 1e-6] = 1.0
    train_tensor = torch.tensor(train_x / scale, dtype=torch.float32, device=device)
    val_tensor = torch.tensor(val_x / scale, dtype=torch.float32, device=device)
    train_target = torch.tensor(train_y, dtype=torch.float32, device=device)
    val_target = torch.tensor(val_y, dtype=torch.float32, device=device)
    train_groups = _group_tensors(train_frame, device, event_bp)
    val_groups = _group_tensors(val_frame, device, event_bp)
    model = HierarchicalResidualNetwork(
        train_x.shape[1], swap_signs=swap_signs
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    best_state = None
    best_loss = float("inf")
    stale = 0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        prediction = model(train_tensor, *train_groups)
        loss = torch.nn.functional.smooth_l1_loss(prediction, train_target)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_loss = float(
                torch.nn.functional.mse_loss(
                    model(val_tensor, *val_groups), val_target
                ).item()
            )
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scale


def train_hierarchical_residual(
    *,
    pairs_path: str | Path,
    sequence_embeddings: str | Path,
    graph_embeddings: str | Path,
    cohort_path: str | Path,
    out_dir: str | Path,
    split_mode: str = "donor",
    target_column: str = "methylation_delta",
    event_bp: int = 1_000_000,
    ridge_alpha: float = 100.0,
    ridge_alpha_grid: tuple[float, ...] | None = None,
    epochs: int = 100,
    patience: int = 12,
    device: str = "cpu",
    seed: int = 42,
    full_window_sequence_control: bool = False,
    validation_shrinkage: bool = False,
    sequence_baseline: str = "ridge",
) -> dict[str, Any]:
    annotation_candidates = [
        "delta_repeat_fraction",
        "delta_hmmflagger_fraction",
        "delta_hmmflagger_dup_fraction",
        "delta_hmmflagger_col_fraction",
        "delta_hmmflagger_err_fraction",
        "delta_hmmflagger_nnn_fraction",
        "delta_copy_number_proxy",
        "delta_graph_mappability_proxy",
        "sv_context_score",
        "sv_length_delta_bp",
        "node_jaccard",
    ]
    required_columns = {
        "donor_id",
        "h1_embedding_id",
        "h2_embedding_id",
        "h1_contig",
        "h1_window_start",
        target_column,
        *annotation_candidates,
    }
    sequence_control_candidates = [
        "delta_gc_fraction",
        "delta_cpg_density",
        "delta_base_entropy",
    ]
    if full_window_sequence_control:
        required_columns.update(sequence_control_candidates)
        # Canonical 3-mer fractions were accumulated across the complete
        # 10-kb path window during exact-path extraction.
        required_columns.update(
            f"delta_kmer3_{kmer}"
            for kmer in (
                "AAA", "AAC", "AAG", "AAT", "ACA", "ACC", "ACG", "ACT",
                "AGA", "AGC", "AGG", "ATA", "ATC", "ATG", "CAA", "CAC",
                "CAG", "CCA", "CCC", "CCG", "CGA", "CGC", "CTA", "CTC",
                "GAA", "GAC", "GCA", "GCC", "GGA", "GTA", "TAA", "TCA",
            )
        )
    pairs = pd.read_csv(
        pairs_path,
        compression="infer",
        low_memory=False,
        usecols=lambda column: column in required_columns,
    )
    cohort = (
        pd.read_csv(cohort_path, sep="\t")
        .drop_duplicates("donor_id")
        [["donor_id", "population", "super_population"]]
    )
    pairs = pairs.drop(
        columns=[
            column
            for column in ("population", "super_population")
            if column in pairs
        ]
    ).merge(cohort, on="donor_id", how="left", validate="many_to_one")
    if pairs["super_population"].isna().any():
        raise ValueError("Some donors lack population metadata.")
    sequence_x = _paired_embedding_deltas(pairs, sequence_embeddings)
    whole_window_sequence_x = np.empty((len(pairs), 0), np.float32)
    sequence_control_columns: list[str] = []
    if full_window_sequence_control:
        sequence_control_columns = [
            column
            for column in pairs.columns
            if column in sequence_control_candidates
            or column.startswith("delta_kmer3_")
        ]
        if len(sequence_control_columns) != 35:
            raise ValueError(
                "Whole-window sequence control requires GC, CpG density, "
                "entropy, and 32 canonical 3-mer delta features."
            )
        whole_window_sequence_x = (
            pairs[sequence_control_columns]
            .fillna(0.0)
            .to_numpy(np.float32)
        )
        sequence_x = np.concatenate(
            [
                sequence_x,
                whole_window_sequence_x,
            ],
            axis=1,
        )
    graph_x = _paired_embedding_deltas(pairs, graph_embeddings)
    annotation_columns = [
        column for column in annotation_candidates if column in pairs
    ]
    annotation_x = (
        pairs[annotation_columns].fillna(0.0).to_numpy(np.float32)
        if annotation_columns
        else np.empty((len(pairs), 0), np.float32)
    )
    residual_feature_sets = {
        "hierarchical_graph_only_residual": graph_x,
        "hierarchical_graph_context_residual": np.concatenate(
            [graph_x, annotation_x], axis=1
        ),
    }
    if annotation_x.shape[1]:
        residual_feature_sets["hierarchical_annotation_only_residual"] = (
            annotation_x
        )
    annotation_swap_signs = np.asarray(
        [
            -1.0 if column.startswith("delta_") else 1.0
            for column in annotation_columns
        ],
        dtype=np.float32,
    )
    residual_swap_signs = {
        "hierarchical_graph_only_residual": -np.ones(
            graph_x.shape[1], dtype=np.float32
        ),
        "hierarchical_graph_context_residual": np.concatenate(
            [
                -np.ones(graph_x.shape[1], dtype=np.float32),
                annotation_swap_signs,
            ]
        ),
    }
    if annotation_x.shape[1]:
        residual_swap_signs["hierarchical_annotation_only_residual"] = (
            annotation_swap_signs
        )
    y = pairs[target_column].to_numpy(np.float32)
    if split_mode == "donor":
        groups = pairs["donor_id"].astype(str).to_numpy()
    elif split_mode == "population":
        groups = pairs["super_population"].astype(str).to_numpy()
    else:
        raise ValueError("split_mode must be 'donor' or 'population'.")
    unique_groups = sorted(set(groups))
    prediction_rows: list[dict[str, Any]] = []
    shrinkage_rows: list[dict[str, Any]] = []
    ridge_selection_rows: list[dict[str, Any]] = []
    if sequence_baseline not in {"ridge", "histgb_whole_window"}:
        raise ValueError(
            "sequence_baseline must be 'ridge' or 'histgb_whole_window'."
        )
    if sequence_baseline == "histgb_whole_window" and not full_window_sequence_control:
        raise ValueError(
            "histgb_whole_window requires --full-window-sequence-control."
        )
    baseline_name = (
        "whole_window_antisymmetric_histgb"
        if sequence_baseline == "histgb_whole_window"
        else (
        (
            "nt_plus_whole_window_sequence_nested_ridge"
            if ridge_alpha_grid
            else "nt_plus_whole_window_sequence_ridge"
        )
        if full_window_sequence_control
        else (
            "frozen_sequence_nested_ridge"
            if ridge_alpha_grid
            else "frozen_sequence_ridge"
        )
        )
    )

    for fold, heldout in enumerate(unique_groups):
        print(
            f"[hierarchical] split={split_mode} fold={fold + 1}/"
            f"{len(unique_groups)} heldout={heldout}",
            flush=True,
        )
        test_idx = np.flatnonzero(groups == heldout)
        available_train_groups = [group for group in unique_groups if group != heldout]
        if len(available_train_groups) < 2:
            raise ValueError("At least three groups are needed for nested validation.")
        validation_group = available_train_groups[fold % len(available_train_groups)]
        val_idx = np.flatnonzero(groups == validation_group)
        train_idx = np.flatnonzero(
            (groups != heldout) & (groups != validation_group)
        )
        if full_window_sequence_control:
            sequence_scaler = StandardScaler().fit(sequence_x[train_idx])
            sequence_train_x = sequence_scaler.transform(sequence_x[train_idx])
            sequence_val_x = sequence_scaler.transform(sequence_x[val_idx])
            sequence_test_x = sequence_scaler.transform(sequence_x[test_idx])
        else:
            sequence_train_x = sequence_x[train_idx]
            sequence_val_x = sequence_x[val_idx]
            sequence_test_x = sequence_x[test_idx]
        if sequence_baseline == "histgb_whole_window":
            sequence_train, sequence_val, sequence_test = (
                _fit_predict_antisymmetric_histgb(
                    whole_window_sequence_x[train_idx],
                    y[train_idx],
                    whole_window_sequence_x[train_idx],
                    whole_window_sequence_x[val_idx],
                    whole_window_sequence_x[test_idx],
                    seed=seed + fold,
                )
            )
        else:
            candidate_alphas = (
                tuple(float(value) for value in ridge_alpha_grid)
                if ridge_alpha_grid
                else (float(ridge_alpha),)
            )
            ridge_sweep: list[dict[str, float]] = []
            fitted_models: dict[float, Ridge] = {}
            for candidate_alpha in candidate_alphas:
                candidate_model = Ridge(
                    alpha=candidate_alpha,
                    fit_intercept=False,
                    solver="lsqr",
                )
                candidate_model.fit(sequence_train_x, y[train_idx])
                candidate_val = candidate_model.predict(sequence_val_x)
                ridge_sweep.append(
                    {
                        "alpha": candidate_alpha,
                        "validation_mae": float(
                            mean_absolute_error(y[val_idx], candidate_val)
                        ),
                    }
                )
                fitted_models[candidate_alpha] = candidate_model
            selected_ridge = min(
                ridge_sweep,
                key=lambda row: (row["validation_mae"], row["alpha"]),
            )
            selected_alpha = float(selected_ridge["alpha"])
            sequence_model = fitted_models[selected_alpha]
            ridge_selection_rows.append(
                {
                    "fold": int(fold),
                    "heldout_group": str(heldout),
                    "validation_group": str(validation_group),
                    "selected_alpha": selected_alpha,
                    "validation_sweep": ridge_sweep,
                }
            )
            sequence_train = sequence_model.predict(sequence_train_x)
            sequence_val = sequence_model.predict(sequence_val_x)
            sequence_test = sequence_model.predict(sequence_test_x)
        val_groups = _group_tensors(
            pairs.iloc[val_idx].reset_index(drop=True), device, event_bp
        )
        test_groups = _group_tensors(
            pairs.iloc[test_idx].reset_index(drop=True), device, event_bp
        )
        residual_predictions: dict[str, np.ndarray] = {}
        for model_name, residual_x in residual_feature_sets.items():
            print(
                f"[hierarchical] heldout={heldout} model={model_name}",
                flush=True,
            )
            residual_model, residual_scale = _fit_residual(
                pairs.iloc[train_idx].reset_index(drop=True),
                pairs.iloc[val_idx].reset_index(drop=True),
                residual_x[train_idx],
                residual_x[val_idx],
                y[train_idx] - sequence_train,
                y[val_idx] - sequence_val,
                device=device,
                event_bp=event_bp,
                seed=seed + fold,
                epochs=epochs,
                patience=patience,
                swap_signs=residual_swap_signs[model_name],
            )
            residual_model.eval()
            test_tensor = torch.tensor(
                residual_x[test_idx] / residual_scale,
                dtype=torch.float32,
                device=device,
            )
            with torch.inference_mode():
                residual_test = (
                    residual_model(test_tensor, *test_groups)
                    .detach()
                    .cpu()
                    .numpy()
                )
            residual_predictions[model_name] = sequence_test + residual_test
            if validation_shrinkage:
                val_tensor = torch.tensor(
                    residual_x[val_idx] / residual_scale,
                    dtype=torch.float32,
                    device=device,
                )
                with torch.inference_mode():
                    residual_val = (
                        residual_model(val_tensor, *val_groups)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                alpha, sweep = _select_residual_shrinkage(
                    y[val_idx],
                    sequence_val,
                    residual_val,
                )
                shrunk_name = model_name.replace(
                    "_residual", "_shrunk_residual"
                )
                residual_predictions[shrunk_name] = (
                    sequence_test + alpha * residual_test
                )
                shrinkage_rows.append(
                    {
                        "fold": int(fold),
                        "heldout_group": str(heldout),
                        "validation_group": str(validation_group),
                        "model": model_name,
                        "shrunk_model": shrunk_name,
                        "selected_alpha": float(alpha),
                        "validation_sweep": sweep,
                    }
                )
        for local, row_index in enumerate(test_idx):
            common = {
                "row_index": int(row_index),
                "fold": fold,
                "heldout_group": heldout,
                "donor_id": str(pairs.iloc[row_index]["donor_id"]),
                "super_population": str(
                    pairs.iloc[row_index]["super_population"]
                ),
                "y_true": float(y[row_index]),
            }
            prediction_rows.append(
                {
                    **common,
                    "model": baseline_name,
                    "y_pred": float(sequence_test[local]),
                }
            )
            prediction_rows.extend(
                {
                    **common,
                    "model": model_name,
                    "y_pred": float(prediction[local]),
                }
                for model_name, prediction in residual_predictions.items()
            )
            prediction_rows.append(
                {**common, "model": "zero", "y_pred": 0.0}
            )

    predictions = pd.DataFrame(prediction_rows)
    metrics = [
        {"model": name, "n": len(frame), **_metrics(frame.y_true, frame.y_pred)}
        for name, frame in predictions.groupby("model", sort=False)
    ]
    per_group = [
        {
            "model": name,
            "heldout_group": group,
            "n": len(frame),
            **_metrics(frame.y_true.to_numpy(), frame.y_pred.to_numpy()),
        }
        for (name, group), frame in predictions.groupby(
            ["model", "heldout_group"], sort=False
        )
    ]
    per_group_frame = pd.DataFrame(per_group)
    metric_names = ("spearman", "mae", "r2", "direction_accuracy")
    macro_metrics = []
    for name, frame in per_group_frame.groupby("model", sort=False):
        row: dict[str, Any] = {
            "model": name,
            "n_groups": int(frame["heldout_group"].nunique()),
        }
        for metric in metric_names:
            finite = frame[metric].to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            row[metric] = float(np.mean(finite)) if len(finite) else float("nan")
        macro_metrics.append(row)
    bootstrap_differences = _paired_group_bootstrap(
        per_group_frame,
        baseline=baseline_name,
        candidate=(
            "hierarchical_graph_context_shrunk_residual"
            if validation_shrinkage
            else "hierarchical_graph_context_residual"
        ),
        seed=seed,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        out_dir / "heldout_predictions.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(metrics).to_csv(out_dir / "metrics.csv", index=False)
    per_group_frame.to_csv(out_dir / "per_group_metrics.csv", index=False)
    pd.DataFrame(macro_metrics).to_csv(out_dir / "macro_metrics.csv", index=False)
    pd.DataFrame(bootstrap_differences).to_csv(
        out_dir / "paired_group_bootstrap_differences.csv", index=False
    )
    if shrinkage_rows:
        shrinkage_table = pd.DataFrame(
            [
                {
                    key: value
                    for key, value in row.items()
                    if key != "validation_sweep"
                }
                for row in shrinkage_rows
            ]
        )
        shrinkage_table.to_csv(
            out_dir / "validation_shrinkage.csv", index=False
        )
    summary = {
        "task": "hierarchical_haplotype_residual",
        "split_mode": split_mode,
        "n_pairs": len(pairs),
        "n_donors": int(pairs["donor_id"].nunique()),
        "n_groups": len(unique_groups),
        "groups": unique_groups,
        "sequence_embeddings": str(sequence_embeddings),
        "sequence_baseline_model": baseline_name,
        "sequence_baseline": sequence_baseline,
        "ridge_alpha": float(ridge_alpha),
        "ridge_alpha_grid": list(ridge_alpha_grid) if ridge_alpha_grid else None,
        "ridge_selection": ridge_selection_rows,
        "full_window_sequence_control": bool(full_window_sequence_control),
        "full_window_sequence_features": sequence_control_columns,
        "graph_embeddings": str(graph_embeddings),
        "annotation_features": annotation_columns,
        "residual_feature_sets": {
            name: int(values.shape[1])
            for name, values in residual_feature_sets.items()
        },
        "residual_swap_antisymmetry": (
            "exact f(Tx)=-f(x), where T negates graph/delta features and "
            "preserves symmetric SV/Jaccard context"
        ),
        "hierarchy": f"10-kb window -> {event_bp}-bp locus event -> donor path",
        "nested_validation": True,
        "validation_only_residual_shrinkage": bool(validation_shrinkage),
        "shrinkage": shrinkage_rows,
        "primary_metric_aggregation": "macro-average across heldout groups",
        "metrics": metrics,
        "macro_metrics": macro_metrics,
        "paired_group_bootstrap_differences": bootstrap_differences,
        "per_group_metrics": per_group,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--sequence-embeddings", required=True)
    parser.add_argument("--graph-embeddings", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split-mode", choices=["donor", "population"], default="donor")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--ridge-alpha-grid",
        type=float,
        nargs="+",
        help="Optional nested-validation grid for the sequence ridge baseline.",
    )
    parser.add_argument("--full-window-sequence-control", action="store_true")
    parser.add_argument("--validation-shrinkage", action="store_true")
    parser.add_argument(
        "--sequence-baseline",
        choices=["ridge", "histgb_whole_window"],
        default="ridge",
    )
    args = parser.parse_args()
    result = train_hierarchical_residual(
        pairs_path=args.pairs,
        sequence_embeddings=args.sequence_embeddings,
        graph_embeddings=args.graph_embeddings,
        cohort_path=args.cohort,
        out_dir=args.out_dir,
        split_mode=args.split_mode,
        device=args.device,
        epochs=args.epochs,
        ridge_alpha_grid=(
            tuple(args.ridge_alpha_grid) if args.ridge_alpha_grid else None
        ),
        full_window_sequence_control=args.full_window_sequence_control,
        validation_shrinkage=args.validation_shrinkage,
        sequence_baseline=args.sequence_baseline,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

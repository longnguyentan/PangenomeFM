"""Recompute the fixed junction pilot's validation metrics and random controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from tasks.entex.prepare import fingerprint
from tasks.transfer.report import save_figure

ARMS = ["incoming_default", "incoming_linear", "bidirectional_default", "bidirectional_linear"]
IDENTITY = ["slice", "target_sn", "closure", "u_local", "v_local", "y_true"]


def validation_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if not set(frame.split).issubset({"train_chr_val", "val_chr_val"}):
        raise ValueError("Pilot contains unexpected held-out or training predictions")
    val = frame.loc[frame.split.eq("val_chr_val")].sort_values(IDENTITY).reset_index(drop=True)
    if (val.empty or val.duplicated(IDENTITY[:-1]).any() or set(val.y_true) != {0, 1}
            or not val.p_edge.between(0, 1).all()):
        raise ValueError("Invalid or duplicate validation candidates")
    digest = hashlib.sha256(val[IDENTITY].to_csv(index=False).encode()).hexdigest()
    return val, digest


def summarize_roots(trained: Path, random: Path) -> tuple[pd.DataFrame, list[dict]]:
    rows, sources = [], []
    declared_arms, sequence_inputs = None, None
    configs = {}
    for root, mode in [(trained, "trained"), (random, "frozen_random")]:
        status = json.loads((root / "status.json").read_text())
        arms = list(status.get("commands", {}))
        if (status.get("status") != "complete" or status.get("heldout_predictions_requested", True)
                or status.get("biological_labels_used", True)
                or not arms or set(arms) - set(ARMS)
                or any(status["returncodes"].get(arm) != 0 for arm in arms)):
            raise ValueError("Incomplete or incorrectly scoped pilot")
        if declared_arms is None:
            declared_arms, sequence_inputs = arms, status.get("sequence_inputs")
        elif arms != declared_arms or status.get("sequence_inputs") != sequence_inputs:
            raise ValueError("Trained/random arms or sequence inputs differ")
        sources.append(fingerprint(root / "status.json"))
        for arm in arms:
            directory = root / arm / "run_001"
            paths = list(directory.glob("*pooled_predictions*.csv.gz"))
            checkpoints = list(directory.glob("ckpt_*.pt"))
            if len(paths) != 1 or len(checkpoints) != 1:
                raise ValueError("Expected exactly one prediction file and checkpoint per arm")
            val, digest = validation_predictions(pd.read_csv(paths[0], float_precision="round_trip"))
            checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
            cfg = checkpoint["args"]
            if (not cfg["validation_only"] or cfg["seed"] != 42
                    or cfg["junction_geometry_match"] != "signed_gap_bins"):
                raise ValueError("Checkpoint configuration differs from declared pilot")
            is_random = mode == "frozen_random"
            if bool(cfg.get("freeze_encoder", False)) != is_random:
                raise ValueError("Checkpoint random/trained attribution mismatch")
            expected_inputs = "cache" if sequence_inputs else "none"
            if cfg.get("node_extra_features", "none") != expected_inputs:
                raise ValueError("Checkpoint node-input attribution mismatch")
            configs[(arm, mode)] = {key: cfg.get(key) for key in [
                "seed", "split_seed", "val_chrs", "test_chrs", "hidden_dim", "n_layers", "n_heads",
                "graph_message_direction", "linear_predictor", "junction_geometry_match",
                "junction_geometry_bin_ratio", "node_structure_source", "node_extra_features",
                "node_feature_cache", "node_feature_min_coverage", "epochs", "patience",
                "lr", "weight_decay", "drop_edge_rate", "mask_query_edges"]}
            if is_random and (not checkpoint.get("initial_encoder_sha256")
                              or checkpoint["initial_encoder_sha256"] != checkpoint["final_encoder_sha256"]):
                raise ValueError("Frozen random encoder weights changed")
            macro = np.mean([roc_auc_score(g.y_true, g.p_edge) for _, g in val.groupby("slice")
                             if g.y_true.nunique() == 2])
            if not np.isclose(macro, checkpoint["best_val_auc"], atol=1e-10, rtol=0):
                raise ValueError("Prediction replay differs from selected checkpoint score")
            rows.append(dict(arm=arm, encoder=mode, n=len(val), n_windows=val.slice.nunique(),
                             positive_prevalence=val.y_true.mean(), auprc=average_precision_score(val.y_true, val.p_edge),
                             auroc=roc_auc_score(val.y_true, val.p_edge), macro_window_auroc=macro,
                             epochs_run=checkpoint["epochs_run"], candidates_sha256=digest,
                             checkpoint_sha256=fingerprint(checkpoints[0])["sha256"],
                             initial_encoder_sha256=checkpoint.get("initial_encoder_sha256"),
                             prediction_sha256=fingerprint(paths[0])["sha256"],
                             representation="sequence_conditioned_graph" if sequence_inputs else "topology_native",
                             evaluation_partition="development_validation"))
    frame = pd.DataFrame(rows)
    if frame.candidates_sha256.nunique() != 1:
        raise ValueError("Candidate identity or labels differ across trained/random arms")
    frame['initialization_check'] = 'passed'
    for arm, group in frame.groupby("arm"):
        missing_initialization = group.initial_encoder_sha256.isna().any()
        if ((sequence_inputs is not None and missing_initialization)
                or (not missing_initialization and group.initial_encoder_sha256.nunique() != 1)
                or configs[(arm, "trained")] != configs[(arm, "frozen_random")]):
            raise ValueError("Trained/random initialization or experiment settings differ")
        if missing_initialization:
            frame.loc[frame.arm.eq(arm), 'initialization_check'] = 'unavailable_in_historical_trained_checkpoint'
    return frame, sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-root", type=Path, required=True)
    parser.add_argument("--random-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    frame, sources = summarize_roots(args.trained_root, args.random_root)
    contrasts = []
    for arm, group in frame.groupby("arm", sort=False):
        indexed = group.set_index("encoder")
        for metric in ["auprc", "auroc", "macro_window_auroc"]:
            contrasts.append(dict(arm=arm, metric=metric,
                                  trained_minus_random=indexed.loc["trained", metric] - indexed.loc["frozen_random", metric]))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.out_dir / "validation_metrics.csv", index=False)
    pd.DataFrame(contrasts).to_csv(args.out_dir / "paired_differences.csv", index=False)
    selected = frame.loc[frame.encoder.eq("trained")].sort_values("macro_window_auroc", ascending=False).iloc[0]
    (args.out_dir / "audit.json").write_text(json.dumps(dict(
        status="complete", source_receipts=sources, candidate_identity="passed",
        frozen_random_parameter_identity="passed", checkpoint_metric_replay="passed",
        matched_initialization=sorted(frame.initialization_check.unique()),
        matched_configuration="passed", representation=selected.representation,
        selected_arm=selected.arm, selection_criterion="native window-macro validation AUROC",
        selected_checkpoint_sha256=selected.checkpoint_sha256,
        scope="single-fold development comparison; not independent biological evidence",
        confidence_intervals="not estimated after selection on this development partition",
    ), indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    arms = frame.arm.drop_duplicates().tolist()
    for mode, color, offset in [("trained", "#245a81", -0.15), ("frozen_random", "#be6831", 0.15)]:
        data = frame.loc[frame.encoder.eq(mode)].set_index("arm").loc[arms]
        ax.bar(np.arange(len(arms)) + offset, data.auprc, width=0.3, label=mode.replace("_", " "), color=color)
    ax.axhline(0.5, color="0.4", ls="--", lw=1, label="Prevalence")
    labels = dict(zip(ARMS, ["Incoming\nMLP", "Incoming\nLinear", "Bidirectional\nMLP", "Bidirectional\nLinear"]))
    ax.set(xticks=range(len(arms)), xticklabels=[labels[arm] for arm in arms],
           ylim=(0, 1), ylabel="Validation AUPRC", title=f"Junction reconstruction: {selected.representation.replace('_', ' ')}")
    ax.legend(frameon=False, loc="upper left")
    save_figure(fig, args.out_dir, "junction_validation")
    plt.close(fig)


if __name__ == "__main__":
    main()

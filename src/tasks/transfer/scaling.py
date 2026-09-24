"""Nested chromosome-balanced window subsets using the exact manuscript trainer."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_full_multicohort_all_chromosomes import Runner, load_config
from scripts.server.run_ccre_frozen_probe_fold import _canonical_chrom
from tasks.entex.prepare import fingerprint


FRACTIONS = (0.125, 0.25, 0.5, 1.0)


def nested_manifests(
    manifest: pd.DataFrame, fold: dict, seed: int = 20260924
) -> dict[float, pd.DataFrame]:
    required = {"name", "target_sn", "start", "end", "closure"}
    if not required.issubset(manifest) or manifest.name.duplicated().any():
        raise ValueError("Invalid original manifest")
    test, validation = set(fold["test"]), set(fold["validation"])
    if test & validation:
        raise ValueError("Overlapping chromosome partitions")
    x = manifest.copy()
    x["_chrom"] = x.target_sn.map(_canonical_chrom)
    held = x._chrom.isin(test | validation)
    windows = x.loc[~held, ["_chrom", "start", "end"]].drop_duplicates()
    if (
        windows.empty
        or not x._chrom.isin(test).any()
        or not x._chrom.isin(validation).any()
    ):
        raise ValueError("Empty chromosome partition")
    windows["rank"] = [
        hashlib.sha256(f"{seed}:{c}:{s}:{e}".encode()).hexdigest()
        for c, s, e in windows.itertuples(index=False, name=None)
    ]
    result, previous = {}, set()
    held_names = set(x.loc[held, "name"])
    for fraction in FRACTIONS:
        chosen = set()
        for _, group in windows.groupby("_chrom", sort=True):
            selected = group.sort_values("rank").head(
                max(1, int(np.ceil(len(group) * fraction)))
            )
            chosen.update(
                selected[["_chrom", "start", "end"]].itertuples(index=False, name=None)
            )
        if not previous.issubset(chosen):
            raise AssertionError("Non-nested training windows")
        keys = list(x[["_chrom", "start", "end"]].itertuples(index=False, name=None))
        keep = held | pd.Series([key in chosen for key in keys], index=x.index)
        selected = x.loc[keep, manifest.columns].copy()
        if (
            set(
                selected.loc[
                    selected.target_sn.map(_canonical_chrom).isin(test | validation),
                    "name",
                ]
            )
            != held_names
        ):
            raise AssertionError("Validation/test universe changed with scale")
        result[fraction] = selected
        previous = chosen
    pd.testing.assert_frame_equal(result[1.0], manifest)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("configs/server_full_multicohort_20260806.json"),
    )
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--checkpoint-root", type=Path, required=True)
    ap.add_argument("--folds", nargs="+", default=["fold_a"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--contexts", nargs="+", default=["strict"])
    ap.add_argument("--smoke-epochs", type=int)
    args = ap.parse_args()
    config = load_config(args.config)
    base = json.loads(Path("configs/entex_v1.json").read_text())
    graph = Path(config["datasets"]["hprc_r2"]["segments"])
    if fingerprint(graph)["sha256"] != base["full_segments_sha256"]:
        raise ValueError("Scaling graph is not the exact manuscript HPRC R2 graph")
    source = pd.read_csv(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    cfg = copy.deepcopy(config)
    cfg["outputs"] = dict(
        root=str(args.checkpoint_root),
        logs=str(args.out_dir / "logs"),
        state=str(args.out_dir / "training_state.json"),
    )
    if args.smoke_epochs is not None:
        if args.smoke_epochs < 1:
            raise ValueError("Smoke epochs must be positive")
        cfg["training"]["epochs"] = args.smoke_epochs
    runner = Runner(cfg, execute=False, phases=set())
    tasks, audits = [], []
    for fold in cfg["rotating_chromosome_folds"]:
        if fold["name"] not in args.folds:
            continue
        subsets = nested_manifests(source, fold)
        for index, (fraction, subset) in enumerate(subsets.items()):
            scale = f"fraction_{fraction:g}"
            directory = args.out_dir / fold["name"] / scale
            directory.mkdir(parents=True)
            path = directory / "manifest.csv"
            subset.to_csv(path, index=False)
            chrom = subset.target_sn.map(_canonical_chrom)
            train = ~chrom.isin(fold["test"] + fold["validation"])
            audits.append(
                dict(
                    fold=fold["name"],
                    fraction=fraction,
                    n_train_manifest_rows=int(train.sum()),
                    n_validation_manifest_rows=int(
                        chrom.isin(fold["validation"]).sum()
                    ),
                    n_test_manifest_rows=int(chrom.isin(fold["test"]).sum()),
                    manifest=fingerprint(path),
                )
            )
            for seed in args.seeds:
                for context in args.contexts:
                    if context not in {"strict", "1hop"}:
                        raise ValueError("Invalid context")
                    out = (
                        args.checkpoint_root
                        / scale
                        / "rotating_folds/hprc_r2"
                        / fold["name"]
                        / f"seed_{seed}"
                        / context
                    )
                    command = runner.training_command(
                        primary="hprc_r2",
                        output_root=str(out),
                        closure=context,
                        seed=seed,
                        test=tuple(fold["test"]),
                        validation=tuple(fold["validation"]),
                    )
                    command[command.index("--manifest") + 1] = str(path)
                    command += ["--canonical_conflict_policy", "exclude"]
                    tasks.append(
                        dict(
                            id=f"scaling_{fold['name']}_{fraction:g}_{seed}_{context}",
                            stage=f"scale_{index}",
                            description="Same graph/model/objective; nested training windows; fixed validation/test",
                            resource="gpu",
                            cost="medium",
                            requires_paths=[str(path), str(graph)],
                            command=command,
                        )
                    )
    if not tasks:
        raise ValueError("No scaling jobs selected")
    (args.out_dir / "roadmap.json").write_text(
        json.dumps(dict(run_name="pretraining_data_scaling_v1", tasks=tasks), indent=2)
        + "\n"
    )
    (args.out_dir / "audit.json").write_text(
        json.dumps(
            dict(
                original_manifest=fingerprint(args.manifest),
                original_config=fingerprint(args.config),
                graph=fingerprint(graph),
                manifests=audits,
                smoke_only=args.smoke_epochs is not None,
                fractions=FRACTIONS,
                sampling_seed=20260924,
                optimization_seeds=args.seeds,
                canonical_conflict_policy="exclude consistently at every scale, including newly trained 100% control",
                meaning="pretraining-window scaling, not population-diversity scaling or a scaling law",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

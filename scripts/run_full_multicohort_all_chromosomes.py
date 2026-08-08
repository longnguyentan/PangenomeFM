#!/usr/bin/env python3
"""Resumable full-cohort/all-chromosome experiment orchestrator.

The runner intentionally separates the final all-data checkpoints from the
rotating chromosome holdouts used for scientific evaluation.  HPRC R1.1 is a
release-transfer dataset, not an additional training cohort, because pooling
it with R2 would duplicate donors and superseded assemblies.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]


def _expand_environment(value):
    """Expand environment variables in every string stored in a JSON config."""
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "${" in expanded:
            raise ValueError(
                f"Unresolved environment variable in configuration value: {value}. "
                "Export the server paths documented in the runbook first."
            )
        return expanded
    return value


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass
class StepRecord:
    name: str
    status: str
    command: list[str]
    started_at: str
    finished_at: str | None = None
    wall_seconds: float | None = None
    log: str | None = None
    outputs: list[str] | None = None
    message: str | None = None


class Runner:
    def __init__(
        self,
        config: dict,
        *,
        execute: bool,
        phases: set[str],
        datasets: set[str] | None = None,
        regimes: set[str] | None = None,
        seeds: set[int] | None = None,
        contexts: set[str] | None = None,
        folds: set[str] | None = None,
        pairs: set[str] | None = None,
        state_file: str | None = None,
    ):
        self.config = config
        self.execute = execute
        self.phases = phases
        self.python = Path(os.environ.get("PYTHON_BIN", sys.executable)).resolve()
        self.device = os.environ.get("DEVICE", config["training"]["device_default"])
        if self.device == "auto":
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        self.root = _path(config["outputs"]["root"])
        self.logs = _path(config["outputs"]["logs"])
        self.state_path = _path(state_file or config["outputs"]["state"])
        self.dataset_filter = datasets
        self.regime_filter = regimes
        self.seed_filter = seeds
        self.context_filter = contexts
        self.fold_filter = folds
        self.pair_filter = pairs
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, StepRecord] = {}
        if self.state_path.exists():
            prior = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.records = {
                name: StepRecord(**record)
                for name, record in prior.get("steps", {}).items()
            }

    def save_state(self) -> None:
        payload = {
            "schema_version": 1,
            "analysis_id": self.config["analysis_id"],
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "python": str(self.python),
            "device": self.device,
            "steps": {name: asdict(record) for name, record in self.records.items()},
        }
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.state_path)

    @staticmethod
    def _glob_exists(patterns: Iterable[str]) -> bool:
        for pattern in patterns:
            # ``Path.parent.glob(path.name)`` only expands wildcards in the
            # final component.  Training outputs are versioned below an
            # intermediate ``run_*`` directory, so that implementation falsely
            # marked successful jobs as failed even when their checkpoints
            # existed.  Expand the complete absolute pattern instead.
            if glob.glob(str(_path(pattern))):
                continue
            return False
        return True

    def run(
        self,
        name: str,
        command: list[str],
        *,
        output_globs: Iterable[str] = (),
        allow_existing: bool = True,
    ) -> None:
        patterns = list(output_globs)
        if allow_existing and patterns and self._glob_exists(patterns):
            self.records[name] = StepRecord(
                name=name,
                status="verified_existing",
                command=command,
                started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                wall_seconds=0.0,
                outputs=patterns,
            )
            self.save_state()
            print(f"[existing] {name}")
            return

        log_path = self.logs / f"{name}.log"
        record = StepRecord(
            name=name,
            status="planned" if not self.execute else "running",
            command=command,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            log=str(log_path),
            outputs=patterns,
        )
        self.records[name] = record
        self.save_state()
        print(f"[{record.status}] {name}: {shlex.join(command)}")
        if not self.execute:
            return

        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get(
            "PYTHONPATH", ""
        )
        started = time.monotonic()
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n$ {shlex.join(command)}\n")
            log.flush()
            process = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record.wall_seconds = time.monotonic() - started
        record.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if process.returncode == 0 and (not patterns or self._glob_exists(patterns)):
            record.status = "completed"
        else:
            record.status = "failed"
            record.message = f"exit_code={process.returncode}"
            self.save_state()
            raise RuntimeError(f"Step failed: {name}; inspect {log_path}")
        self.save_state()

    def preflight(self) -> None:
        usage = shutil.disk_usage(REPO_ROOT)
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "python": str(self.python),
            "device": self.device,
            "disk_total_bytes": usage.total,
            "disk_used_bytes": usage.used,
            "disk_free_bytes": usage.free,
            "datasets": {},
        }
        for name, dataset in self.config["datasets"].items():
            files = {}
            for key in ("raw_gfa", "segments", "links"):
                if key in dataset:
                    path = _path(dataset[key])
                    files[key] = {
                        "path": dataset[key],
                        "exists": path.exists(),
                        "bytes": path.stat().st_size if path.exists() else None,
                    }
            payload["datasets"][name] = files
        out = self.root / "preflight.json"
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self.records["preflight"] = StepRecord(
            name="preflight",
            status="completed",
            command=[],
            started_at=payload["timestamp"],
            finished_at=payload["timestamp"],
            wall_seconds=0.0,
            outputs=[str(out)],
        )
        self.save_state()

    def selected_datasets(self):
        for name, dataset in self.config["datasets"].items():
            if self.dataset_filter is None or name in self.dataset_filter:
                yield name, dataset

    def parse_graphs(self) -> None:
        datasets = self.config["datasets"]
        for name, ds in self.selected_datasets():
            if not ds.get("raw_gfa"):
                segments = _path(ds["segments"])
                links = _path(ds["links"])
                if not (segments.exists() and links.exists()):
                    raise FileNotFoundError(
                        f"{name} has no raw_gfa and its parsed tables are missing: "
                        f"{segments}, {links}"
                    )
                continue
            self.run(
                f"parse_{name}",
                [
                    str(self.python),
                    "-m",
                    "graphgenomefm",
                    "parse-gfa",
                    "--gfa",
                    ds["raw_gfa"],
                    "--out-dir",
                    ds["data_dir"],
                ],
                output_globs=[
                    f"{ds['data_dir']}/parse_summary.json",
                    ds["segments"],
                    ds["links"],
                ],
            )

    def validate_graphs(self) -> None:
        for name, ds in self.selected_datasets():
            validation = f"{self.config['outputs']['validation']}/{name}"
            command = [
                str(self.python),
                "-m",
                "graph.validation",
                "--segments",
                ds["segments"],
                "--links",
                ds["links"],
                "--out-dir",
                validation,
                "--label",
                name,
                "--max-node-length",
                "1024",
                "--sequence-sample-modulus",
                "10",
            ]
            if ds.get("path_metadata") and _path(ds["path_metadata"]).exists():
                command.extend(["--path-metadata", ds["path_metadata"]])
            self.run(
                f"validate_{name}",
                command,
                output_globs=[
                    f"{validation}/validation_summary.json",
                    f"{validation}/validation_issues.csv",
                ],
            )

    def index_gbz_paths(self) -> None:
        for name, ds in self.selected_datasets():
            if not ds.get("raw_gbz") or not ds.get("path_metadata"):
                continue
            metadata = ds["path_metadata"]
            self.run(
                f"index_paths_{name}",
                [
                    str(self.python),
                    str(REPO_ROOT / "scripts/server/index_gbz_paths.py"),
                    "--gbz",
                    ds["raw_gbz"],
                    "--output",
                    metadata,
                ],
                output_globs=[metadata, f"{metadata}.summary.json"],
            )

    def build_benchmarks(self, *, full_coverage: bool) -> None:
        datasets = dict(self.selected_datasets())
        if full_coverage:
            full = self.config["full_coverage_pretraining"]
            for name, ds in datasets.items():
                self.run(
                    f"benchmark_full_coverage_{name}",
                    [
                        str(self.python),
                        "-m",
                        "graphgenomefm",
                        "make-benchmark",
                        "--data-dir",
                        ds["data_dir"],
                        "--out-dir",
                        ds["pretrain_benchmark"],
                        "--targets",
                        "all",
                        "--target-prefix",
                        ds["reference_prefix"],
                        "--closures",
                        *full["closures"],
                        "--window-bp",
                        str(full["window_bp"]),
                        "--n-windows",
                        "1",
                        "--tile-stride-bp",
                        str(full["tile_stride_bp"]),
                        "--negative-sampler",
                        full["negative_sampler"],
                        "--negative-coord-band",
                        str(full["negative_coord_band"]),
                        "--negative-tol-bp",
                        str(full["negative_tolerance_bp"]),
                        "--negative-tol-frac",
                        str(full["negative_tolerance_fraction"]),
                        "--negative-degree-matched",
                        "--negative-shortfall-policy",
                        full.get("negative_shortfall_policy", "paired_subsample"),
                        "--matched-closure-windows",
                        "--seed",
                        str(full["seed"]),
                        "--no-network-analysis",
                        "--no-viz",
                    ],
                    output_globs=[f"{ds['pretrain_benchmark']}/manifest.csv"],
                )
            return

        bench = self.config["benchmark"]
        for name, ds in datasets.items():
            self.run(
                f"benchmark_{name}",
                [
                    str(self.python),
                    "-m",
                    "graphgenomefm",
                    "make-benchmark",
                    "--data-dir",
                    ds["data_dir"],
                    "--out-dir",
                    ds["benchmark"],
                    "--targets",
                    "all",
                    "--target-prefix",
                    ds["reference_prefix"],
                    "--closures",
                    *bench["closures"],
                    "--window-bp",
                    str(bench["window_bp"]),
                    "--n-windows",
                    str(bench["windows_per_chromosome"]),
                    "--negative-sampler",
                    bench["negative_sampler"],
                    "--negative-coord-band",
                    str(bench["negative_coord_band"]),
                    "--negative-tol-bp",
                    str(bench["negative_tolerance_bp"]),
                    "--negative-tol-frac",
                    str(bench["negative_tolerance_fraction"]),
                    "--negative-degree-matched",
                    "--non-overlapping-windows",
                    "--matched-closure-windows",
                    "--seed",
                    str(bench["seed"]),
                    "--no-network-analysis",
                    "--no-viz",
                ],
                output_globs=[f"{ds['benchmark']}/manifest.csv"],
            )

    def prepare(self) -> None:
        self.parse_graphs()
        self.index_gbz_paths()
        self.validate_graphs()
        self.build_benchmarks(full_coverage=False)
        self.build_benchmarks(full_coverage=True)

    def selected_values(self, key: str, selected: set | None):
        values = self.config["training"][key]
        return [value for value in values if selected is None or value in selected]

    def selected_regimes(self, matrix_key: str, defaults: list[dict]):
        regimes = self.config.get("experiment_matrix", {}).get(matrix_key, defaults)
        for regime in regimes:
            if self.regime_filter is None or regime["name"] in self.regime_filter:
                yield regime

    def selected_pairs(self, matrix_key: str, defaults: list[dict]):
        pairs = self.config.get("experiment_matrix", {}).get(matrix_key, defaults)
        for pair in pairs:
            name = pair.get("name", f"{pair['source']}_to_{pair['target']}")
            if self.pair_filter is None or name in self.pair_filter:
                yield {**pair, "name": name}

    def training_command(
        self,
        *,
        primary: str,
        output_root: str,
        closure: str,
        seed: int,
        extras: tuple[str, ...] = (),
        test: tuple[str, ...] = (),
        validation: tuple[str, ...] = (),
        domain_adversarial: bool = False,
        resume_recovery: str | None = None,
    ) -> list[str]:
        cfg = self.config["training"]
        ds = self.config["datasets"][primary]
        command = [
            str(self.python),
            "-m",
            "training.pretrain",
            "--manifest",
            f"{ds['pretrain_benchmark']}/manifest.csv",
            "--full_segments",
            ds["segments"],
            "--out_dir",
            output_root,
            "--primary_dataset_name",
            primary,
            "--closures",
            closure,
            "--hidden_dim",
            str(cfg["hidden_dim"]),
            "--n_heads",
            str(cfg["heads"]),
            "--n_layers",
            str(cfg["layers"]),
            "--epochs",
            str(cfg["epochs"]),
            "--patience",
            str(cfg["patience"]),
            "--seed",
            str(seed),
            "--split_seed",
            str(cfg["split_seed"]),
            "--device",
            self.device,
            "--dual_stream",
            "--adaptive_window",
            "--adaptive_window_base",
            "32",
            "--adaptive_window_alpha",
            "4.0",
            "--multiscale_rope",
            "--n_rope_scales",
            "3",
            "--orientation_rope",
            "--focal_loss",
            "--focal_gamma",
            "2.0",
            "--drop_edge",
            "--drop_edge_rate",
            "0.1",
            "--warmup_epochs",
            "5",
            "--mask_query_edges",
            "--save_predictions",
            "--recovery_every",
            str(cfg["recovery_every_epochs"]),
            "--batch_size",
            str(cfg["candidate_mask_batch_size"]),
            "--lazy_tensorize",
        ]
        if extras:
            command.append("--extra_datasets")
            for extra in extras:
                extra_ds = self.config["datasets"][extra]
                command.extend(
                    [
                        extra,
                        f"{extra_ds['pretrain_benchmark']}/manifest.csv",
                        extra_ds["segments"],
                    ]
                )
        if test:
            command.extend(["--test_chrs", *test])
        if validation:
            command.extend(["--val_chrs", *validation])
        if domain_adversarial:
            command.extend(
                [
                    "--domain_adversarial",
                    "--domain_loss_weight",
                    "0.1",
                    "--domain_grl_lambda",
                    "1.0",
                ]
            )
        if resume_recovery:
            command.extend(["--resume_recovery", resume_recovery])
        return command

    @staticmethod
    def latest_recovery(output_root: str, closure: str) -> str | None:
        candidates = sorted(
            _path(output_root).glob(f"run_*/recovery_{closure}__*.pt")
        )
        return str(candidates[-1]) if candidates else None

    def train_final_models(self) -> None:
        root = self.config["outputs"]["root"]
        defaults = [
            {"name": "hprc_r2", "primary": "hprc_r2", "extras": []},
            {"name": "hgsvc3", "primary": "hgsvc3", "extras": []},
            {"name": "combined_hprc_r2_hgsvc3", "primary": "hprc_r2", "extras": ["hgsvc3"]},
            {"name": "combined_hprc_r2_hgsvc3_dann", "primary": "hprc_r2", "extras": ["hgsvc3"], "domain_adversarial": True},
        ]
        for spec in self.selected_regimes("final_regimes", defaults):
            regime, primary = spec["name"], spec["primary"]
            extras = tuple(spec.get("extras", []))
            for seed in self.selected_values("seeds", self.seed_filter):
                for closure in self.selected_values("contexts", self.context_filter):
                    out = f"{root}/final_models/{regime}/seed_{seed}/{closure}"
                    self.run(
                        f"train_final_{regime}_seed{seed}_{closure}",
                        self.training_command(
                            primary=primary,
                            output_root=out,
                            closure=closure,
                            seed=seed,
                            extras=extras,
                            domain_adversarial=bool(spec.get("domain_adversarial", False)),
                            resume_recovery=self.latest_recovery(out, closure),
                        ),
                        output_globs=[f"{out}/run_*/ckpt_{closure}__*.pt"],
                    )

    def train_rotating_folds(self) -> None:
        root = self.config["outputs"]["root"]
        defaults = [
            {"name": "hprc_r2", "primary": "hprc_r2", "extras": []},
            {"name": "hgsvc3", "primary": "hgsvc3", "extras": []},
            {"name": "combined_hprc_r2_hgsvc3", "primary": "hprc_r2", "extras": ["hgsvc3"]},
        ]
        for fold in self.config["rotating_chromosome_folds"]:
            if self.fold_filter is not None and fold["name"] not in self.fold_filter:
                continue
            for spec in self.selected_regimes("fold_regimes", defaults):
                regime, primary = spec["name"], spec["primary"]
                extras = tuple(spec.get("extras", []))
                for seed in self.selected_values("seeds", self.seed_filter):
                    for closure in self.selected_values("contexts", self.context_filter):
                        out = (
                            f"{root}/rotating_folds/{regime}/{fold['name']}/"
                            f"seed_{seed}/{closure}"
                        )
                        self.run(
                            f"train_{regime}_{fold['name']}_seed{seed}_{closure}",
                            self.training_command(
                                primary=primary,
                                output_root=out,
                                closure=closure,
                                seed=seed,
                                extras=extras,
                                test=tuple(fold["test"]),
                                validation=tuple(fold["validation"]),
                                resume_recovery=self.latest_recovery(out, closure),
                            ),
                            output_globs=[f"{out}/run_*/ckpt_{closure}__*.pt"],
                        )

    def transfer(self) -> None:
        root = self.config["outputs"]["root"]
        defaults = [
            {"source": "hprc_r2", "target": "hgsvc3"},
            {"source": "hgsvc3", "target": "hprc_r2"},
        ]
        for pair in self.selected_pairs("cohort_transfer_pairs", defaults):
            source, target = pair["source"], pair["target"]
            target_ds = self.config["datasets"][target]
            model_regime = pair.get("model_regime", source)
            for seed in self.selected_values("seeds", self.seed_filter):
                for closure in self.selected_values("contexts", self.context_filter):
                    model_dir = (
                        _path(root)
                        / "final_models"
                        / model_regime
                        / f"seed_{seed}"
                        / closure
                    )
                    checkpoints = sorted(model_dir.glob(f"run_*/ckpt_{closure}__*.pt"))
                    if not checkpoints:
                        if self.execute:
                            raise FileNotFoundError(
                                f"No final checkpoint for {source}, seed={seed}, {closure}"
                            )
                        checkpoint_arg = f"<checkpoint:{source}:seed={seed}:{closure}>"
                    else:
                        checkpoint_arg = str(checkpoints[-1])
                    out = f"{root}/cohort_transfer/{source}_to_{target}/seed_{seed}/{closure}"
                    self.run(
                        f"transfer_{source}_to_{target}_seed{seed}_{closure}",
                        [
                            str(self.python),
                            "-m",
                            "evaluation.external",
                            "--checkpoint",
                            checkpoint_arg,
                            "--manifest",
                            f"{target_ds['benchmark']}/manifest.csv",
                            "--full_segments",
                            target_ds["segments"],
                            "--out_dir",
                            out,
                            "--closure",
                            closure,
                            "--split",
                            "all",
                            "--seed",
                            str(seed),
                            "--dataset_name",
                            target,
                            "--device",
                            self.device,
                            "--mask_query_edges",
                            "--mask_batch_size",
                            str(self.config["training"]["candidate_mask_batch_size"]),
                        ],
                        output_globs=[f"{out}/run_*/summary.json"],
                    )

    def release_transfer(self) -> None:
        root = self.config["outputs"]["root"]
        defaults = [
            {"source": "hprc_r1_1", "target": "hprc_r2"},
            {"source": "hprc_r2", "target": "hprc_r1_1"},
        ]
        for pair in self.selected_pairs("release_transfer_pairs", defaults):
            source, target = pair["source"], pair["target"]
            for seed in self.selected_values("seeds", self.seed_filter):
                for closure in self.selected_values("contexts", self.context_filter):
                    reuse_regime = pair.get("reuse_final_regime")
                    if reuse_regime:
                        model_out = f"{root}/final_models/{reuse_regime}/seed_{seed}/{closure}"
                    else:
                        model_out = f"{root}/release_models/{source}/seed_{seed}/{closure}"
                        self.run(
                            f"train_release_{source}_seed{seed}_{closure}",
                            self.training_command(
                                primary=source,
                                output_root=model_out,
                                closure=closure,
                                seed=seed,
                                resume_recovery=self.latest_recovery(model_out, closure),
                            ),
                            output_globs=[f"{model_out}/run_*/ckpt_{closure}__*.pt"],
                        )
                    checkpoints = sorted(
                        _path(model_out).glob(f"run_*/ckpt_{closure}__*.pt")
                    )
                    if not checkpoints:
                        if self.execute:
                            raise FileNotFoundError(model_out)
                        continue
                    target_ds = self.config["datasets"][target]
                    out = f"{root}/release_transfer/{source}_to_{target}/seed_{seed}/{closure}"
                    self.run(
                        f"release_transfer_{source}_to_{target}_seed{seed}_{closure}",
                        [
                            str(self.python), "-m", "evaluation.external",
                            "--checkpoint", str(checkpoints[-1]),
                            "--manifest", f"{target_ds['benchmark']}/manifest.csv",
                            "--full_segments", target_ds["segments"],
                            "--out_dir", out,
                            "--closure", closure,
                            "--split", "all",
                            "--seed", str(seed),
                            "--dataset_name", target,
                            "--device", self.device,
                            "--mask_query_edges",
                            "--mask_batch_size",
                            str(self.config["training"]["candidate_mask_batch_size"]),
                        ],
                        output_globs=[f"{out}/run_*/summary.json"],
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/full_multicohort_all_chromosomes_20260806.json",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=[
            "prepare", "parse", "paths", "validate", "benchmark", "pretrain_benchmark",
            "final", "folds", "transfer", "release",
        ],
        default=["prepare", "final", "folds", "transfer", "release"],
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute commands. Without this flag, record and print the complete queue.",
    )
    parser.add_argument("--datasets", nargs="+", help="Preparation dataset names.")
    parser.add_argument("--regimes", nargs="+", help="Final/fold regime names.")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--contexts", nargs="+")
    parser.add_argument("--folds", nargs="+")
    parser.add_argument("--pairs", nargs="+", help="Configured transfer pair names.")
    parser.add_argument(
        "--state-file",
        help="Override execution-state path (required for parallel shards).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _expand_environment(
        json.loads(_path(args.config).read_text(encoding="utf-8"))
    )
    runner = Runner(
        config,
        execute=args.execute,
        phases=set(args.phases),
        datasets=set(args.datasets) if args.datasets else None,
        regimes=set(args.regimes) if args.regimes else None,
        seeds=set(args.seeds) if args.seeds else None,
        contexts=set(args.contexts) if args.contexts else None,
        folds=set(args.folds) if args.folds else None,
        pairs=set(args.pairs) if args.pairs else None,
        state_file=args.state_file,
    )
    runner.preflight()
    if "prepare" in runner.phases:
        runner.prepare()
    else:
        if "parse" in runner.phases:
            runner.parse_graphs()
        if "paths" in runner.phases:
            runner.index_gbz_paths()
        if "validate" in runner.phases:
            runner.validate_graphs()
        if "benchmark" in runner.phases:
            runner.build_benchmarks(full_coverage=False)
        if "pretrain_benchmark" in runner.phases:
            runner.build_benchmarks(full_coverage=True)
    if "final" in runner.phases:
        runner.train_final_models()
    if "folds" in runner.phases:
        runner.train_rotating_folds()
    if "transfer" in runner.phases:
        runner.transfer()
    if "release" in runner.phases:
        runner.release_transfer()
    print(f"Execution state: {runner.state_path}")


if __name__ == "__main__":
    main()

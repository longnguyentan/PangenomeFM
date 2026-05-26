from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.gfa_parser import parse_gfa_to_tables
from data.layout import DatasetLayout, find_dataset
from pipeline import default_results_dir, run_module


SEGMENT_REQUIRED = {"id", "name", "seq", "LN", "SN", "SO", "SR"}
LINK_REQUIRED = {"from_seg", "from_orient", "to_seg", "to_orient", "overlap"}
CANONICAL_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def _latest_node_labels(ccre_dir: Path) -> Path:
    candidates = sorted(ccre_dir.glob("run_*/node_labels.csv.gz"))
    if candidates:
        return candidates[-1]
    direct = ccre_dir / "node_labels.csv.gz"
    if direct.exists():
        return direct
    raise FileNotFoundError(
        f"Could not find node_labels.csv.gz under {ccre_dir}. "
        "Run `python -m graphgenomefm label-ccre ...` first or pass --node-labels."
    )


def _read_columns(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=2, compression="infer").columns)


def _format_prefixed_chrom(chrom: str, prefix: str | None) -> str:
    if not prefix:
        return chrom
    if "{}" in prefix:
        return prefix.format(chrom)
    if prefix.endswith(("#", "|", ":", "/", "_", "-")):
        return f"{prefix}{chrom}"
    return f"{prefix}#{chrom}"


def _expand_chroms(chroms: list[str] | None, prefix: str | None) -> list[str] | None:
    if chroms is None:
        return None
    expanded: list[str] = []
    for chrom in chroms:
        if chrom.lower() in {"all", "all-canonical", "canonical"}:
            expanded.extend(_format_prefixed_chrom(c, prefix) for c in CANONICAL_CHROMS)
        elif chrom.lower() in {"ccre-paper", "paper-ccre"}:
            expanded.extend(_format_prefixed_chrom(c, prefix) for c in ["chr16", "chr8", "chr19", "chr22"])
        elif "#" in chrom or "|" in chrom or chrom.startswith("id="):
            expanded.append(chrom)
        else:
            expanded.append(_format_prefixed_chrom(chrom, prefix))
    return expanded


def _print_dataset(dataset: DatasetLayout) -> None:
    print(f"dataset:       {dataset.name}")
    print(f"data_dir:      {dataset.data_dir}")
    print(f"segments:      {dataset.segments}")
    print(f"links:         {dataset.links}")
    print(f"benchmark_dir: {dataset.benchmark_dir}")
    print(f"ccre_dir:      {dataset.ccre_dir}")


def check_data(args: argparse.Namespace) -> int:
    dataset = find_dataset(args.data_dir)
    _print_dataset(dataset)

    seg_cols = set(_read_columns(dataset.segments))
    link_cols = set(_read_columns(dataset.links))
    seg_missing = SEGMENT_REQUIRED - seg_cols
    link_missing = LINK_REQUIRED - link_cols

    print(f"segments_size: {dataset.segments.stat().st_size:,} bytes")
    print(f"links_size:    {dataset.links.stat().st_size:,} bytes")
    print(f"segments_cols: {sorted(seg_cols)}")
    print(f"links_cols:    {sorted(link_cols)}")

    if seg_missing or link_missing:
        if seg_missing:
            print(f"segments missing columns: {sorted(seg_missing)}")
        if link_missing:
            print(f"links missing columns: {sorted(link_missing)}")
        return 2
    print("status:        ok")
    return 0


def parse_gfa(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_sns = _expand_chroms(args.target_sns, args.target_prefix)
    stats = parse_gfa_to_tables(
        gfa_path=args.gfa,
        segments_out=out_dir / "full_segments.csv.gz",
        links_out=out_dir / "full_links.csv.gz",
        summary_out=out_dir / "parse_summary.json",
        max_lines=args.max_lines,
        target_sns=set(target_sns) if target_sns else None,
        include_link_neighbors=args.include_link_neighbors,
    )
    print(f"parsed dataset directory: {out_dir}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    return 0


def make_benchmark(args: argparse.Namespace) -> int:
    dataset = find_dataset(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else dataset.benchmark_dir
    module_args = {
        "data_dir": dataset.data_dir,
        "out_dir": out_dir,
        "segments": dataset.segments.name,
        "links": dataset.links.name,
        "seed": args.seed,
        "window_bp": args.window_bp,
        "n_windows": args.n_windows,
        "targets": _expand_chroms(args.targets, args.target_prefix),
        "closures": args.closures,
        "negative_sampler": args.negative_sampler,
        "allow_cross_sn_negatives": args.allow_cross_sn_negatives,
        "negative_coord_band": args.negative_coord_band,
        "negative_tol_bp": args.negative_tol_bp,
        "negative_tol_frac": args.negative_tol_frac,
        "negative_degree_matched": args.negative_degree_matched,
        "no_network_analysis": args.no_network_analysis,
        "no_viz": args.no_viz,
    }
    print("building benchmark from cleaned dataset directory")
    _print_dataset(dataset)
    run_module("data.make_benchmark", module_args)
    return 0


def pretrain(args: argparse.Namespace) -> int:
    dataset = find_dataset(args.data_dir)
    benchmark_dir = Path(args.benchmark_dir) if args.benchmark_dir else dataset.benchmark_dir
    manifest = benchmark_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"Missing benchmark manifest: {manifest}. "
            "Run `python -m graphgenomefm make-benchmark --data-dir ...` first."
        )
    out_dir = Path(args.out_dir) if args.out_dir else default_results_dir(dataset.name, "pretrain")
    module_args = {
        "manifest": manifest,
        "full_segments": dataset.segments,
        "out_dir": out_dir,
        "test_chrs": _expand_chroms(args.test_chrs, args.target_prefix),
        "val_chrs": _expand_chroms(args.val_chrs, args.target_prefix),
        "hidden_dim": args.hidden_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
        "device": args.device,
        "dual_stream": True,
        "adaptive_window": True,
        "adaptive_window_base": 32,
        "adaptive_window_alpha": 4.0,
        "multiscale_rope": True,
        "n_rope_scales": 3,
        "orientation_rope": True,
        "focal_loss": True,
        "focal_gamma": 2.0,
        "drop_edge": True,
        "drop_edge_rate": 0.1,
        "warmup_epochs": 5,
        "stream_mode": args.stream_mode,
        "no_fusion_gate": args.no_fusion_gate,
    }
    print("starting shared pretraining")
    _print_dataset(dataset)
    print(f"manifest:      {manifest}")
    print(f"out_dir:       {out_dir}")
    run_module("training.pretrain", module_args)
    return 0


def label_ccre(args: argparse.Namespace) -> int:
    dataset = find_dataset(args.data_dir)
    encode_bed = Path(args.encode_bed)
    if not encode_bed.exists():
        raise FileNotFoundError(f"Missing ENCODE cCRE BED: {encode_bed}")
    out_dir = Path(args.out_dir) if args.out_dir else dataset.ccre_dir
    module_args = {
        "full_segments": dataset.segments,
        "ccre_bed": encode_bed,
        "out_dir": out_dir,
        "verbose": args.verbose,
    }
    print("mapping ENCODE cCRE intervals to graph nodes")
    _print_dataset(dataset)
    print(f"encode_bed:    {encode_bed}")
    print(f"out_dir:       {out_dir}")
    run_module("tasks.ccre.label_nodes", module_args)
    return 0


def ccre_baseline(args: argparse.Namespace) -> int:
    dataset = find_dataset(args.data_dir)
    node_labels = Path(args.node_labels) if args.node_labels else _latest_node_labels(dataset.ccre_dir)
    out_dir = Path(args.out_dir) if args.out_dir else default_results_dir(dataset.name, "ccre_baseline")
    module_args = {
        "full_segments": dataset.segments,
        "full_links": dataset.links,
        "node_labels": node_labels,
        "test_chrs": args.test_chrs,
        "val_chr": args.val_chr,
        "out_dir": out_dir,
        "seed": args.seed,
        "downsample_background": args.downsample_background,
    }
    print("running cCRE node baseline")
    _print_dataset(dataset)
    print(f"node_labels:   {node_labels}")
    print(f"out_dir:       {out_dir}")
    run_module("tasks.ccre.baselines", module_args)
    return 0


def ccre_binary_baseline(args: argparse.Namespace) -> int:
    dataset = find_dataset(args.data_dir)
    node_labels = Path(args.node_labels) if args.node_labels else _latest_node_labels(dataset.ccre_dir)
    out_dir = Path(args.out_dir) if args.out_dir else default_results_dir(dataset.name, "ccre_binary_baseline")
    module_args = {
        "full_segments": dataset.segments,
        "full_links": dataset.links,
        "node_labels": node_labels,
        "test_chrs": args.test_chrs,
        "val_chr": args.val_chr,
        "out_dir": out_dir,
        "seed": args.seed,
        "negative_train_fraction": args.negative_train_fraction,
    }
    print("running binary cCRE baseline")
    _print_dataset(dataset)
    print(f"node_labels:   {node_labels}")
    print(f"out_dir:       {out_dir}")
    run_module("tasks.ccre.binary", module_args)
    return 0


def ccre_gat(args: argparse.Namespace) -> int:
    dataset = find_dataset(args.data_dir)
    benchmark_dir = Path(args.benchmark_dir) if args.benchmark_dir else dataset.benchmark_dir
    manifest = benchmark_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"Missing benchmark manifest: {manifest}. "
            "Run `python -m graphgenomefm make-benchmark --data-dir ...` first."
        )
    node_labels = Path(args.node_labels) if args.node_labels else _latest_node_labels(dataset.ccre_dir)
    out_dir = Path(args.out_dir) if args.out_dir else default_results_dir(dataset.name, "ccre_gat")
    module_args = {
        "manifest": manifest,
        "full_segments": dataset.segments,
        "node_labels": node_labels,
        "out_dir": out_dir,
        "task": args.task,
        "test_chrs": _expand_chroms(args.test_chrs, args.target_prefix),
        "val_chrs": _expand_chroms(args.val_chrs, args.target_prefix),
        "hidden_dim": args.hidden_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
        "device": args.device,
        "pretrained_checkpoint": Path(args.pretrained_checkpoint) if args.pretrained_checkpoint else None,
        "freeze_backbone": args.freeze_backbone,
        "keep_is_grch38": args.keep_is_grch38,
        "dual_stream": True,
        "adaptive_window": True,
        "adaptive_window_base": 32,
        "adaptive_window_alpha": 4.0,
        "multiscale_rope": True,
        "n_rope_scales": 3,
        "orientation_rope": True,
        "stream_mode": args.stream_mode,
        "no_fusion_gate": args.no_fusion_gate,
    }
    print("running cCRE GAT")
    _print_dataset(dataset)
    print(f"manifest:      {manifest}")
    print(f"node_labels:   {node_labels}")
    print(f"out_dir:       {out_dir}")
    run_module("tasks.ccre.train_gat", module_args)
    return 0


def ccre_aligned_baseline(args: argparse.Namespace) -> int:
    dataset = find_dataset(args.data_dir)
    node_labels = Path(args.node_labels) if args.node_labels else _latest_node_labels(dataset.ccre_dir)
    out_dir = Path(args.out_dir) if args.out_dir else default_results_dir(dataset.name, "ccre_aligned_baseline")
    module_args = {
        "full_segments": dataset.segments,
        "full_links": dataset.links,
        "node_labels": node_labels,
        "out_dir": out_dir,
        "test_chrs": args.test_chrs,
        "val_chrs": args.val_chrs,
        "method": args.method,
        "feature_set": args.feature_set,
        "label_scheme": args.label_scheme,
        "positive_group": args.positive_group,
        "all_ccre_as_negative": args.all_ccre_as_negative,
        "evaluation_universe": args.evaluation_universe,
        "benchmark_manifest": Path(args.benchmark_manifest) if args.benchmark_manifest else None,
        "closures": args.closures,
        "seed": args.seed,
    }
    print("running aligned cCRE baseline")
    _print_dataset(dataset)
    print(f"node_labels:   {node_labels}")
    print(f"out_dir:       {out_dir}")
    run_module("tasks.ccre.aligned_baselines", module_args)
    return 0


def ccre_embedding_baseline(args: argparse.Namespace) -> int:
    dataset = find_dataset(args.data_dir)
    benchmark_dir = Path(args.benchmark_dir) if args.benchmark_dir else dataset.benchmark_dir
    manifest = benchmark_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing benchmark manifest: {manifest}")
    node_labels = Path(args.node_labels) if args.node_labels else _latest_node_labels(dataset.ccre_dir)
    out_dir = Path(args.out_dir) if args.out_dir else default_results_dir(dataset.name, "ccre_embedding_baseline")
    module_args = {
        "checkpoint": Path(args.checkpoint),
        "manifest": manifest,
        "full_segments": dataset.segments,
        "node_labels": node_labels,
        "out_dir": out_dir,
        "test_chrs": args.test_chrs,
        "val_chrs": args.val_chrs,
        "method": args.method,
        "label_scheme": args.label_scheme,
        "positive_group": args.positive_group,
        "all_ccre_as_negative": args.all_ccre_as_negative,
        "closure": args.closure,
        "device": args.device,
        "seed": args.seed,
        "max_slices": args.max_slices,
        "save_embeddings": args.save_embeddings,
    }
    print("running frozen-embedding cCRE baseline")
    _print_dataset(dataset)
    print(f"manifest:      {manifest}")
    print(f"checkpoint:    {args.checkpoint}")
    print(f"node_labels:   {node_labels}")
    print(f"out_dir:       {out_dir}")
    run_module("tasks.ccre.embedding_baseline", module_args)
    return 0


def eval_external(args: argparse.Namespace) -> int:
    dataset = find_dataset(args.data_dir)
    benchmark_dir = Path(args.benchmark_dir) if args.benchmark_dir else dataset.benchmark_dir
    manifest = benchmark_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"Missing benchmark manifest: {manifest}. "
            "Run `python -m graphgenomefm make-benchmark --data-dir ...` first."
        )
    out_dir = Path(args.out_dir) if args.out_dir else default_results_dir(dataset.name, "external_eval")
    module_args = {
        "checkpoint": Path(args.checkpoint),
        "manifest": manifest,
        "full_segments": dataset.segments,
        "out_dir": out_dir,
        "closure": args.closure,
        "split": args.split,
        "seed": args.seed,
        "device": args.device,
    }
    print("running frozen external link-prediction evaluation")
    _print_dataset(dataset)
    print(f"manifest:      {manifest}")
    print(f"checkpoint:    {args.checkpoint}")
    print(f"out_dir:       {out_dir}")
    run_module("evaluation.external", module_args)
    return 0


def impute_edges(args: argparse.Namespace) -> int:
    dataset = find_dataset(args.data_dir)
    benchmark_dir = Path(args.benchmark_dir) if args.benchmark_dir else dataset.benchmark_dir
    manifest = benchmark_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing benchmark manifest: {manifest}")
    out_dir = Path(args.out_dir) if args.out_dir else default_results_dir(dataset.name, "edge_imputation")
    module_args = {
        "checkpoint": Path(args.checkpoint),
        "manifest": manifest,
        "full_segments": dataset.segments,
        "out_dir": out_dir,
        "closure": args.closure,
        "split": args.split,
        "candidate_label": args.candidate_label,
        "top_k": args.top_k,
        "comparison_segments": Path(args.comparison_segments) if args.comparison_segments else None,
        "comparison_links": Path(args.comparison_links) if args.comparison_links else None,
        "device": args.device,
        "seed": args.seed,
    }
    print("scoring candidate missing edges for graph imputation")
    _print_dataset(dataset)
    print(f"manifest:      {manifest}")
    print(f"checkpoint:    {args.checkpoint}")
    print(f"out_dir:       {out_dir}")
    run_module("evaluation.impute_edges", module_args)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="graphgenomefm",
        description="Simple dataset-directory CLI for GraphGenome-FM.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check-data", help="Validate data/<dataset>/full_segments and full_links files.")
    p.add_argument("--data-dir", required=True)
    p.set_defaults(func=check_data)

    p = sub.add_parser("parse-gfa", help="Convert a raw GFA/rGFA file to a cleaned dataset directory.")
    p.add_argument("--gfa", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-lines", type=int, default=None)
    p.add_argument("--target-sns", nargs="+", default=None)
    p.add_argument("--target-prefix", default=None)
    p.add_argument("--include-link-neighbors", action="store_true")
    p.set_defaults(func=parse_gfa)

    p = sub.add_parser("make-benchmark", help="Build link-prediction benchmark slices.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window-bp", type=int, default=50_000)
    p.add_argument("--n-windows", type=int, default=10)
    p.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help=(
            "Target SNs/chromosomes. Short names use --target-prefix. "
            "Use 'all' for chr1-22,chrX,chrY or 'ccre-paper' for chr16/8/19/22."
        ),
    )
    p.add_argument("--target-prefix", default="GRCh38#0")
    p.add_argument("--closures", nargs="+", default=["strict", "1hop"], choices=["strict", "1hop"])
    p.add_argument(
        "--negative-sampler",
        default="random",
        choices=["random", "hard_coord_degree", "distance_matched"],
        help="Negative sampler for generated edge_pred files.",
    )
    p.add_argument(
        "--allow-cross-sn-negatives",
        action="store_true",
        help="For hard samplers, allow negative endpoints to span different SN/chromosome values.",
    )
    p.add_argument("--negative-coord-band", type=int, default=5_000)
    p.add_argument("--negative-tol-bp", type=int, default=1_000)
    p.add_argument("--negative-tol-frac", type=float, default=0.10)
    p.add_argument("--negative-degree-matched", action="store_true")
    p.add_argument("--no-network-analysis", action="store_true")
    p.add_argument("--no-viz", action="store_true")
    p.set_defaults(func=make_benchmark)

    p = sub.add_parser("pretrain", help="Run shared graph foundation-model pretraining.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--benchmark-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--test-chrs", nargs="+", default=["chr1", "chr8", "chr19", "chrY"])
    p.add_argument("--val-chrs", nargs="+", default=["chr16"])
    p.add_argument("--target-prefix", default="GRCh38#0")
    p.add_argument("--hidden-dim", type=int, default=48)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--stream-mode", choices=["full", "coordinate", "graph"], default="full")
    p.add_argument("--no-fusion-gate", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.set_defaults(func=pretrain)

    p = sub.add_parser("label-ccre", help="Map ENCODE cCRE BED intervals to graph nodes.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--encode-bed", default="data/encode/GRCh38-human-cCREs.bed")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--verbose", action="store_true", default=True)
    p.set_defaults(func=label_ccre)

    p = sub.add_parser("ccre-baseline", help="Run structural-feature cCRE baseline.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--node-labels", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--test-chrs", nargs="+", default=["chr8", "chr19", "chr22"])
    p.add_argument("--val-chr", default="chr16")
    p.add_argument("--downsample-background", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=ccre_baseline)

    p = sub.add_parser("ccre-binary-baseline", help="Run binary cCRE/non-cCRE baseline.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--node-labels", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--test-chrs", nargs="+", default=["chr8", "chr19", "chr22"])
    p.add_argument("--val-chr", default="chr16")
    p.add_argument("--negative-train-fraction", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=ccre_binary_baseline)

    p = sub.add_parser("ccre-gat", help="Run graph neural cCRE classifier.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--benchmark-dir", default=None)
    p.add_argument("--node-labels", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument(
        "--task",
        default="multiclass",
        choices=["multiclass", "full9", "binary", "group3", "group4", "group5"],
    )
    p.add_argument("--test-chrs", nargs="+", default=["chr8", "chr19", "chr22"])
    p.add_argument("--val-chrs", nargs="+", default=["chr16"])
    p.add_argument("--target-prefix", default="GRCh38#0")
    p.add_argument("--hidden-dim", type=int, default=48)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--stream-mode", choices=["full", "coordinate", "graph"], default="full")
    p.add_argument("--no-fusion-gate", action="store_true")
    p.add_argument("--pretrained-checkpoint", default=None)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--keep-is-grch38", action="store_true")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.set_defaults(func=ccre_gat)

    p = sub.add_parser("ccre-aligned-baseline", help="Run aligned all-node/window cCRE baselines.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--node-labels", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--test-chrs", nargs="+", default=["chr8", "chr19", "chr22"])
    p.add_argument("--val-chrs", nargs="+", default=["chr16"])
    p.add_argument("--method", choices=["logistic", "mlp", "random_forest"], default="logistic")
    p.add_argument(
        "--feature-set",
        choices=["coordinate", "graph", "structural", "linearized_graph"],
        default="structural",
    )
    p.add_argument(
        "--label-scheme",
        choices=["binary", "full9", "multiclass", "group3", "group4", "group5", "category_binary"],
        default="binary",
    )
    p.add_argument("--positive-group", default=None)
    p.add_argument("--all-ccre-as-negative", action="store_true")
    p.add_argument(
        "--evaluation-universe",
        choices=["all", "benchmark_windows"],
        default="all",
    )
    p.add_argument("--benchmark-manifest", default=None)
    p.add_argument("--closures", nargs="+", choices=["strict", "1hop"], default=["strict", "1hop"])
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=ccre_aligned_baseline)

    p = sub.add_parser("ccre-embedding-baseline", help="Run cCRE baselines on frozen GraphGenome-FM embeddings.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--benchmark-dir", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--node-labels", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--test-chrs", nargs="+", default=["chr8", "chr19", "chr22"])
    p.add_argument("--val-chrs", nargs="+", default=["chr16"])
    p.add_argument("--method", choices=["logistic", "mlp", "random_forest"], default="logistic")
    p.add_argument(
        "--label-scheme",
        choices=["binary", "full9", "multiclass", "group3", "group4", "group5", "category_binary"],
        default="binary",
    )
    p.add_argument("--positive-group", default=None)
    p.add_argument("--all-ccre-as-negative", action="store_true")
    p.add_argument("--closure", choices=["strict", "1hop", "all"], default="strict")
    p.add_argument("--max-slices", type=int, default=None)
    p.add_argument("--save-embeddings", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.set_defaults(func=ccre_embedding_baseline)

    p = sub.add_parser("eval-external", help="Evaluate a frozen pretraining checkpoint on an external benchmark.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--benchmark-dir", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--closure", default=None, choices=["strict", "1hop", "all"])
    p.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.set_defaults(func=eval_external)

    p = sub.add_parser("impute-edges", help="Score candidate missing edges for HGSVC graph imputation.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--benchmark-dir", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--closure", default="1hop", choices=["strict", "1hop", "all"])
    p.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    p.add_argument("--candidate-label", type=int, default=0)
    p.add_argument("--top-k", type=int, default=1000)
    p.add_argument("--comparison-segments", default=None)
    p.add_argument("--comparison-links", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.set_defaults(func=impute_edges)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

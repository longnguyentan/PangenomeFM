"""Generate frozen transfer jobs for the repository's existing roadmap executor."""

import argparse
import json
from pathlib import Path

from scripts.server.run_ccre_frozen_probe_matrix import build_jobs
from tasks.transfer.scaling import FRACTIONS


def scaling_probe_tasks(
    jobs: list, checkpoint_root: Path, out_root: Path
) -> list[dict]:
    tasks = []
    for fraction in FRACTIONS:
        scale = f"fraction_{fraction:g}"
        for job in jobs:
            tasks.append(
                dict(
                    id=f"scaling_probe_{fraction:g}_{job.fold}_{job.seed}_{job.closure}",
                    stage=f"batch_{len(tasks) % 8}",
                    resource="cpu",
                    cost="medium",
                    description="Frozen SV/cCRE/CTCF probes using identical biological data at every scale",
                    requires_paths=[str(checkpoint_root / scale)],
                    command=[
                        "python",
                        "-m",
                        "tasks.transfer.scaling_evaluate",
                        "--checkpoint-root",
                        str(checkpoint_root / scale),
                        "--out-root",
                        str(out_root / scale),
                        "--fold",
                        job.fold,
                        "--seed",
                        str(job.seed),
                        "--context",
                        job.closure,
                        "--device",
                        "cpu",
                        "--execute",
                    ],
                )
            )
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-plan", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--kind", choices=["hg008", "scaling-probes", "hr-controls"], default="hg008")
    ap.add_argument("--checkpoint-root", type=Path)
    ap.add_argument("--main-checkout", type=Path)
    ap.add_argument("--topology-control-cache", type=Path)
    args = ap.parse_args()
    config = json.loads(Path("configs/entex_v1.json").read_text())
    manuscript = json.loads(Path(config["manuscript_config"]).read_text())
    jobs = build_jobs(manuscript)
    if args.kind == "hr-controls":
        if args.main_checkout is None or args.topology_control_cache is None:
            ap.error("H/R controls require --main-checkout and --topology-control-cache")
        pairs = sorted({(job.fold, job.seed) for job in jobs})
        tasks = [dict(
            id=f"hr_{fold}_{seed}", stage=f"worker_{index % 2}", resource="gpu", cost="medium",
            description="Frozen trained/random SV and cCRE controls in both contexts; all original folds retained as exploratory reuse",
            requires_paths=[str(args.main_checkout), str(args.topology_control_cache)],
            command=["python", "scripts/server/run_v2_review_controls.py",
                     "--main-checkout", str(args.main_checkout),
                     "--topology-control-cache", str(args.topology_control_cache),
                     "--fold", fold, "--seed", str(seed),
                     "--out-root", str(args.out_root / fold / f"seed_{seed}")],
        ) for index, (fold, seed) in enumerate(pairs)]
        args.out_plan.parent.mkdir(parents=True, exist_ok=True)
        args.out_plan.write_text(json.dumps(dict(
            schema_version=1, run_name="hr_controls_20260927", tasks=tasks,
            scope="Exploratory retrospective controls; these chromosome folds were previously inspected",
            initialization_seeds=sorted({seed for _, seed in pairs}),
            encoder_policy="Frozen trained and random; no biological label gradients",
        ), indent=2) + "\n")
        return
    if args.kind == "scaling-probes":
        if args.checkpoint_root is None:
            ap.error("--checkpoint-root is required for scaling probes")
        tasks = scaling_probe_tasks(jobs, args.checkpoint_root, args.out_root)
        args.out_plan.parent.mkdir(parents=True, exist_ok=True)
        args.out_plan.write_text(
            json.dumps(
                dict(
                    schema_version=1, run_name="scaling_frozen_biology_v1", tasks=tasks
                ),
                indent=2,
            )
            + "\n"
        )
        return
    examples = "data/downstream_v2/v1/hg008_mapping/sv_breakpoint_examples.csv.gz"
    tasks = []
    for i, job in enumerate(jobs):
        tasks.append(
            dict(
                id=f"hg008_{job.fold}_{job.seed}_{job.closure}",
                stage=f"batch_{i % 4}",
                description="Reconstruct original HGSVC probe, check regression, score excluded-chromosome HG008",
                resource="cpu",
                cost="medium",
                depends_on=[],
                requires_paths=[examples],
                command=[
                    "python",
                    "-m",
                    "tasks.transfer.external_sv",
                    "--external-examples",
                    examples,
                    "--out-root",
                    str(args.out_root),
                    "--fold",
                    job.fold,
                    "--seed",
                    str(job.seed),
                    "--context",
                    job.closure,
                ],
            )
        )
    args.out_plan.parent.mkdir(parents=True, exist_ok=True)
    args.out_plan.write_text(
        json.dumps(
            dict(schema_version=1, run_name="hg008_transfer_v1", tasks=tasks), indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

"""Generate HG008 jobs for the repository's existing roadmap executor."""

import argparse
import json
from pathlib import Path

from scripts.server.run_ccre_frozen_probe_matrix import build_jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-plan", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    args = ap.parse_args()
    config = json.loads(Path("configs/entex_v1.json").read_text())
    manuscript = json.loads(Path(config["manuscript_config"]).read_text())
    jobs = build_jobs(manuscript)
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

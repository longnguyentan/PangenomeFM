#!/usr/bin/env python3
"""Build evidence-safe submission tables from the downloaded full-run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BASELINES = [
    {
        "baseline": "Full dual-stream topology encoder",
        "input_access": "coordinates + graph topology; no nucleotide tokens",
        "implemented": True,
        "executed_on_identical_rotating_folds": True,
        "same_examples_and_splits": True,
        "evidence": "120 rotating-fold jobs: 4 regimes x 5 folds x 3 seeds x 2 contexts",
        "submission_use": "primary model",
        "required_action": "none for topology claim",
    },
    {
        "baseline": "Coordinate-only neural encoder",
        "input_access": "coordinates only",
        "implemented": True,
        "executed_on_identical_rotating_folds": False,
        "same_examples_and_splits": False,
        "evidence": "pretrain CLI stream_mode=coordinate plus 150-job identical-fold ablation config; execution pending",
        "submission_use": "required ablation",
        "required_action": "execute configs/server_rotating_ablation_hprc_20260809.json",
    },
    {
        "baseline": "Graph-only neural encoder",
        "input_access": "graph topology only",
        "implemented": True,
        "executed_on_identical_rotating_folds": False,
        "same_examples_and_splits": False,
        "evidence": "pretrain CLI stream_mode=graph plus 150-job identical-fold ablation config; execution pending",
        "submission_use": "required ablation",
        "required_action": "execute configs/server_rotating_ablation_hprc_20260809.json",
    },
    {
        "baseline": "No fusion gate",
        "input_access": "same coordinate and graph inputs as full model",
        "implemented": True,
        "executed_on_identical_rotating_folds": False,
        "same_examples_and_splits": False,
        "evidence": "pretrain CLI --no_fusion_gate plus identical-fold ablation config; execution pending",
        "submission_use": "required ablation",
        "required_action": "execute configs/server_rotating_ablation_hprc_20260809.json",
    },
    {
        "baseline": "No coordinate/multiscale/orientation positional encoding",
        "input_access": "same topology with default positional options disabled",
        "implemented": True,
        "executed_on_identical_rotating_folds": False,
        "same_examples_and_splits": False,
        "evidence": "ablation explicitly uses --no_rope and omits multiscale/orientation RoPE; execution pending",
        "submission_use": "required ablation",
        "required_action": "execute configs/server_rotating_ablation_hprc_20260809.json",
    },
    {
        "baseline": "No path encoding",
        "input_access": "not applicable",
        "implemented": False,
        "executed_on_identical_rotating_folds": False,
        "same_examples_and_splits": False,
        "evidence": "current encoder has no donor-path encoder to ablate",
        "submission_use": "not an ablation of this model",
        "required_action": "mark N/A; do not imply that path encoding was tested",
    },
    {
        "baseline": "Simple topology heuristics",
        "input_access": "degree/common-neighbor/local topology",
        "implemented": True,
        "executed_on_identical_rotating_folds": False,
        "same_examples_and_splits": True,
        "evidence": "implemented in evaluate_rotating_link_baselines.py with per-positive query masking and fixed split seed; execution pending",
        "submission_use": "required non-neural baseline",
        "required_action": "execute run_rotating_baseline_matrix.py",
    },
    {
        "baseline": "Coordinate/distance SGD logistic classifier",
        "input_access": "endpoint coordinates and distances",
        "implemented": True,
        "executed_on_identical_rotating_folds": False,
        "same_examples_and_splits": True,
        "evidence": "implemented with training-chromosome-only fitting and validation-chromosome-only calibration; execution pending",
        "submission_use": "required shortcut baseline",
        "required_action": "execute run_rotating_baseline_matrix.py",
    },
    {
        "baseline": "Sequence-composition edge classifier",
        "input_access": "endpoint mono/di-nucleotide composition with no graph topology",
        "implemented": True,
        "executed_on_identical_rotating_folds": False,
        "same_examples_and_splits": True,
        "evidence": "implemented as streaming SGD on identical candidates/folds with validation-only calibration; execution pending",
        "submission_use": "required sequence-access baseline",
        "required_action": "execute composition baseline; add frozen DNA-LM baseline for journal scope",
    },
    {
        "baseline": "Random/frequency predictor",
        "input_access": "training-fold class frequency only",
        "implemented": True,
        "executed_on_identical_rotating_folds": False,
        "same_examples_and_splits": True,
        "evidence": "frequency and seeded uniform predictions are emitted by identical-fold baseline evaluator; execution pending",
        "submission_use": "required sanity baseline",
        "required_action": "execute run_rotating_baseline_matrix.py",
    },
    {
        "baseline": "Frozen DNA-language-model edge features",
        "input_access": "endpoint nucleotide embeddings",
        "implemented": False,
        "executed_on_identical_rotating_folds": False,
        "same_examples_and_splits": False,
        "evidence": "Nucleotide Transformer is supported for methylation, not topology-link candidates",
        "submission_use": "journal-strength baseline",
        "required_action": "cache endpoint embeddings once, fit identical chromosome-fold probes",
    },
]


LEAKAGE_ROWS = [
    ("Positive query edge visible to message passing", "controlled", "query edges are removed before message passing", "masked reconstruction claim is supported"),
    ("Chromosome overlap", "controlled in primary rotating folds", "five folds cover chr1-chr22, chrX, chrY", "use rotating-fold metrics as primary"),
    ("All-data final-model evaluation", "not a generalization test", "training and evaluation use the aggregate all-data graph", "label as checkpoint production only"),
    ("Donor/haplotype leakage inside aggregate graph", "uncontrolled", "examples are induced from pooled topology rather than donor paths", "no donor- or haplotype-held-out claim"),
    ("HPRC R2/HGSVC3 donor overlap", "uncontrolled", "path metadata verifies four shared donors: HG00733, HG02818, NA19036, and NA19240", "cross-graph is not strictly independent-cohort transfer"),
    ("Coordinate interval mismatch between contexts", "controlled", "240 coordinate-matched pairs per 50 kb dataset", "does not control graph exposure"),
    ("Node/link/candidate exposure between contexts", "uncontrolled", "zero exposure-matched pairs under the preregistered 10% node/link caliper", "treat contexts as different tasks"),
    ("Candidate split randomness", "controlled", "candidate split seed fixed at 20260806 across model seeds", "seed variance reflects model initialization"),
    ("Negative-sampling coordinate shortcut", "partially controlled", "distance- and degree-matched negatives", "residual topology shortcuts remain possible"),
    ("Nucleotide or reverse-complement leakage", "not applicable to current topology model", "model consumes no nucleotide tokens", "do not call it a joint sequence-graph model"),
    ("Threshold selection", "controlled by new analysis", "temperature and F1 threshold fit only on val_chr_test", "thresholded metrics may be shown as secondary"),
    ("Probability calibration", "improved but imperfect", "validation-only temperature scaling lowers ECE/NLL/Brier but held-out ECE remains nonzero", "show calibration and retain ranking metrics as primary"),
]


CLAIM_ROWS = [
    ("Query-edge-masked local topology is learnable across all canonical chromosomes", "strongly supported", "five rotating folds, 24 chromosomes, three seeds, HPRC R2 and HGSVC3", "primary claim"),
    ("A shared HPRC R2 plus HGSVC3 model improves within-graph reconstruction", "supported within aggregate graphs", "shared-model AUPRC is 0.980/0.992 on HPRC and 0.979/0.991 on HGSVC3 for strict/one-hop", "state effect size; do not imply donor independence"),
    ("Representations transfer across released aggregate graphs", "moderately supported", "frozen HPRC-to-HGSVC AUPRC 0.600/0.941 and reverse 0.725/0.890", "use directional cross-graph wording"),
    ("HGSVC3 is intrinsically weaker than HPRC R2", "contradicted", "standalone HGSVC3 exceeds HPRC R2 on strict chromosome-held-out AUPRC", "remove; discuss directional transfer and construction"),
    ("One-hop context improves representation independently of exposure", "unsupported", "zero pairs satisfy joint 10% node/link exposure caliper", "report strict and expanded as different tasks"),
    ("Transfer is independent of donor", "unsupported", "four shared donors: HG00733, HG02818, NA19036, NA19240", "rematerialize graphs without shared assemblies"),
    ("PangenomeFM is a general-purpose biological foundation model", "unsupported", "one topology objective; no executed genome-wide biological task from these checkpoints", "use pretrained pangenome-topology encoder"),
    ("Graph embeddings add meaningful haplotype-methylation prediction", "unsupported", "graph-only residual Spearman gain 0.00004 with CI crossing zero", "retain as negative pilot"),
    ("Graph divergence associates with H1/H2 methylation divergence on chr8", "preliminary association", "donor-macro Spearman 0.1704 across 10 donors", "do not call predictive or causal"),
]


RISK_ROWS = [
    ("Expanded-context advantage may be exposure-driven", "critical", "zero exposure matches; expanded contexts expose about 1.6x nodes and 1.9x links", "retain separate-task language"),
    ("No donor/haplotype-held-out topology", "critical", "aggregate graphs pool paths before example construction", "materialize path-specific examples and rebuild filtered graphs"),
    ("Foundation-model terminology exceeds evidence", "critical", "only topology reconstruction and aggregate-graph transfer are complete", "use topology-pretraining wording"),
    ("High within-graph scores may reflect shortcuts", "high", "identical-fold ablation/baseline suite is implemented but not executed", "complete 150 GPU and 30 CPU jobs"),
    ("Cross-graph transfer shares donors", "high", "four shared donors are exact path-metadata matches", "exclude assemblies before graph construction"),
    ("Calibration remains imperfect", "high", "temperature scaling lowers ECE but HPRC strict ECE remains 0.142", "keep ranking metrics primary"),
    ("chrY is a failure/stress chromosome", "moderate", "HPRC strict chrY AUPRC 0.868", "report and stratify rather than omit"),
    ("Biological validation is disconnected from full checkpoints", "critical", "cross-fitted cCRE probe implemented but not executed", "execute before journal scope"),
    ("SV truth absent from executed model evaluation", "high", "versioned VCF downloader and streaming audit exist only", "download, audit, then evaluate class/length/AF/breakpoint strata"),
]


def build_exclusion_table(results_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted((results_root / "context_audit").glob("*/context_statistics.csv")):
        dataset = path.parent.name
        frame = pd.read_csv(path)
        excluded = frame.loc[~frame["default_loader_eligible"].astype(bool)].copy()
        excluded.insert(0, "dataset", dataset)
        excluded["exclusion_stage"] = "training loader"
        excluded["exclusion_reason"] = "fewer_than_10_default_training_candidates"
        rows.append(
            excluded[
                [
                    "dataset",
                    "slice_id",
                    "target_sn",
                    "chromosome",
                    "split",
                    "closure_legacy_label",
                    "context_regime",
                    "start",
                    "end",
                    "visible_nodes",
                    "visible_link_rows",
                    "candidate_targets",
                    "default_training_candidates",
                    "default_validation_candidates",
                    "default_test_candidates",
                    "exclusion_stage",
                    "exclusion_reason",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def downstream_evidence(repo_root: Path) -> pd.DataFrame:
    root = repo_root / "results/hprc/haplotype_methylation_chr8_cohort_v2"
    nonlinear = json.loads((root / "nonlinear_incremental_residual_v1/summary.json").read_text())
    biological = json.loads((root / "biological_signal_strong_sequence_v2/summary.json").read_text())
    metrics = {row["model"]: row for row in nonlinear["macro_metrics"]}
    comparisons = {
        (row["candidate"], row["metric"]): row
        for row in nonlinear["paired_donor_comparisons"]
    }
    sequence = metrics["whole_window_antisymmetric_histgb"]
    graph_annotation = metrics["nonlinear_graph_annotation_residual_shrunk"]
    graph_only = metrics["nonlinear_graph_residual_shrunk"]
    association = biological["structural_biological_association"]
    return pd.DataFrame(
        [
            {
                "task": "chr8 within-donor H1/H2 methylation difference",
                "evaluation": "nested donor-held-out",
                "n_donors": nonlinear["n_donors"],
                "n_paired_loci": nonlinear["n_pairs"],
                "model_or_test": "strong whole-window sequence baseline",
                "metric": "donor-macro Spearman",
                "estimate": sequence["spearman"],
                "ci95_low": None,
                "ci95_high": None,
                "interpretation": "strongest baseline; supersedes the weak NT-ridge comparison",
            },
            {
                "task": "chr8 within-donor H1/H2 methylation difference",
                "evaluation": "nested donor-held-out",
                "n_donors": nonlinear["n_donors"],
                "n_paired_loci": nonlinear["n_pairs"],
                "model_or_test": "sequence + graph + annotation residual",
                "metric": "Spearman gain over strong sequence",
                "estimate": comparisons[("nonlinear_graph_annotation_residual_shrunk", "spearman")]["difference"],
                "ci95_low": comparisons[("nonlinear_graph_annotation_residual_shrunk", "spearman")]["ci95_lower"],
                "ci95_high": comparisons[("nonlinear_graph_annotation_residual_shrunk", "spearman")]["ci95_upper"],
                "interpretation": "statistically consistent but biologically small incremental gain",
            },
            {
                "task": "chr8 within-donor H1/H2 methylation difference",
                "evaluation": "nested donor-held-out",
                "n_donors": nonlinear["n_donors"],
                "n_paired_loci": nonlinear["n_pairs"],
                "model_or_test": "sequence + graph-only residual",
                "metric": "Spearman gain over strong sequence",
                "estimate": comparisons[("nonlinear_graph_residual_shrunk", "spearman")]["difference"],
                "ci95_low": comparisons[("nonlinear_graph_residual_shrunk", "spearman")]["ci95_lower"],
                "ci95_high": comparisons[("nonlinear_graph_residual_shrunk", "spearman")]["ci95_upper"],
                "interpretation": "no reliable incremental rank gain from learned graph embeddings alone",
            },
            {
                "task": "graph divergence versus absolute H1/H2 methylation difference",
                "evaluation": "locked post-hoc descriptive donor bootstrap",
                "n_donors": biological["n_donors"],
                "n_paired_loci": biological["n_pairs"],
                "model_or_test": "graph-divergence association",
                "metric": "donor-macro Spearman",
                "estimate": association["donor_macro_spearman_abs_delta_vs_graph_divergence"],
                "ci95_low": association["spearman_ci95_lower"],
                "ci95_high": association["spearman_ci95_upper"],
                "interpretation": "replicated biological association; not proof of predictive model utility",
            },
        ]
    )


def tier_status() -> pd.DataFrame:
    rows = [
        ("before_any_submission", "rotating chromosome results primary", "complete", "all 120 rotating jobs complete"),
        ("before_any_submission", "evidence-safe topology wording", "complete in readiness report and revised manuscript", "foundation-model wording remains explicitly qualified"),
        ("before_any_submission", "complete disclosures", "complete in leakage table", "zero exposure matches, aggregate donor leakage, four shared donors, no nucleotide inputs, calibration disclosed"),
        ("before_any_submission", "baseline access and exact exclusions", "complete for 50 kb evaluation benchmark", "full-coverage 5 Mb generator exclusions were not included in downloaded bundle"),
        ("conference", "validation-only calibration", "complete", "120/120 runs calibrated without held-out tuning"),
        ("conference", "identical-fold ablation/baseline suite", "implemented; not executed", "150 GPU ablation jobs and 30 CPU baseline jobs are configured and tested"),
        ("conference", "exposure- and target-matched contexts", "separate-task interpretation adopted", "zero matches at 10%; no superiority language is allowed"),
        ("computational_genomics_journal", "donor-held-out path-resolved examples", "split manifests and bounded positive-edge materializer implemented; server execution pending", "227 non-overlap HPRC donors/454 haplotypes and 61 HGSVC donors/122 haplotypes; valid absent-path negatives remain unimplemented"),
        ("computational_genomics_journal", "genome-wide cCRE or methylation from downloaded checkpoints", "implemented; server execution pending", "five-fold chromosome-held-out frozen cCRE probe, identical-node baselines, sequence cache, and validation-only calibration are tested"),
        ("computational_genomics_journal", "versioned phased SV truth", "download, streaming audit, breakpoint mapping, and frozen-probe matrix implemented; server execution pending", "HPRC R2 wave VCF and HGSVC3 v1.0 SV/inversion VCFs; insertion/deletion class, length, frequency, chromosome, and mapping strata"),
        ("computational_genomics_journal", "transfer excluding shared donors", "exact overlap, path-union graph rematerializer, and 42-job config implemented; server execution pending", "four donors verified; training is gated on an audit proving the source GFA embeds paths and the derived graph has no missing endpoints"),
        ("nature_methods", "multiple independent biological tasks and cohorts", "not met", "current methylation pilot has negligible graph-only incremental gain"),
        ("nature_methods", "sequence role", "not met", "topology model has no nucleotide encoder"),
        ("nature_methods", "SV and QTL/GWAS evidence", "not met; pipelines implemented", "real chr8 eQTL/sQTL signals normalized; matched-set enrichment and GWAS snapshot normalizer tested; no adequately powered model-prioritized matched-region result"),
        ("nature_methods", "multi-axis transfer and representation scaling", "not met; capacity matrix implemented", "chromosome/cohort/release executed; donor/path and 24x1/96x4/192x6 capacity follow-ups await server execution; ancestry absent"),
    ]
    return pd.DataFrame(rows, columns=["tier", "requirement", "verified_status", "evidence_or_blocker"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    results_root = args.results_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = pd.DataFrame(BASELINES)
    leakage = pd.DataFrame(
        LEAKAGE_ROWS,
        columns=["risk", "status", "evidence", "manuscript_consequence"],
    )
    exclusions = build_exclusion_table(results_root)
    exclusion_summary = (
        exclusions.groupby(["dataset", "closure_legacy_label", "split"], dropna=False)
        .size()
        .rename("excluded_slices")
        .reset_index()
    )
    downstream = downstream_evidence(repo_root)
    tiers = tier_status()
    claims = pd.DataFrame(
        CLAIM_ROWS,
        columns=["claim", "evidence_class", "verified_evidence", "required_wording_or_action"],
    )
    risks = pd.DataFrame(
        RISK_ROWS,
        columns=["reviewer_risk", "severity", "evidence", "mitigation"],
    )

    baseline.to_csv(out_dir / "baseline_access_table.csv", index=False)
    leakage.to_csv(out_dir / "leakage_disclosure_table.csv", index=False)
    exclusions.to_csv(out_dir / "exact_50kb_loader_exclusions.csv", index=False)
    exclusion_summary.to_csv(out_dir / "exact_50kb_loader_exclusion_summary.csv", index=False)
    downstream.to_csv(out_dir / "downstream_evidence_table.csv", index=False)
    tiers.to_csv(out_dir / "submission_tier_status.csv", index=False)
    claims.to_csv(out_dir / "claim_evidence_matrix.csv", index=False)
    risks.to_csv(out_dir / "reviewer_risk_table.csv", index=False)

    calibration_audit = json.loads(
        (args.calibration_dir / "calibration_audit.json").read_text()
    )
    summary = {
        "schema_version": 1,
        "baseline_rows": len(baseline),
        "leakage_rows": len(leakage),
        "exact_50kb_loader_exclusions": len(exclusions),
        "exclusion_counts_by_dataset": exclusions.groupby("dataset").size().to_dict(),
        "downstream_evidence_rows": len(downstream),
        "claim_rows": len(claims),
        "reviewer_risk_rows": len(risks),
        "calibration": calibration_audit,
        "known_missing_artifact": (
            "Full-coverage 5 Mb benchmark exclusions.csv and coverage_report.json were "
            "not packaged in the downloaded server bundle; recover them from server_workspace/data/benchmarks."
        ),
    }
    (out_dir / "readiness_table_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Explicit launch after reviewing the completed three-seed development gate."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[3]
os.chdir(root)
base = root / "results/foundation_evidence_20260927"
checkpoints = list((base / "frozen_branch_seed_replication/seed_314159/sequence_trained").glob("*/run_001/ckpt_*.pt"))
if len(checkpoints) != 1:
    raise ValueError("Expected one previously audited architecture template")
command = [sys.executable, "-u", "scripts/server/run_masked_chromosome_replication.py",
    "--config", "configs/masked_nt_chromosome_replication_20260928.json",
    "--template-checkpoint", str(checkpoints[0]),
    "--main-checkout", "/home/tuv43532/PangenomeFM",
    "--topology-control-cache", "/home/tuv43532/PangenomeFM_review_20260927/results/v2_review_20260927/controls/topology_control.npz",
    "--out-root", str(base / "masked_chromosome_replication"),
    "--reviewed-development-sha256", "3e05b7840c26b328300db07b758eb70a979f677626b17cff771a0218eaf8d6a2",
    "--cache-dependency", str(base / "whole_graph_nt_completion_20260928/status.json"),
    "--wait-hours", "6", "--gpus", "0", "1", "2", "3"]
record = dict(native_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              command=command, review=json.loads((Path(__file__).parent / "development_review.json").read_text()))
(Path(__file__).parent / "launch.json").write_text(json.dumps(record, indent=2)+"\n")
subprocess.run(command, env=dict(os.environ, PYTHONPATH="src:."), check=True)

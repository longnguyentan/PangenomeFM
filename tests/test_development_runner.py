import hashlib
import json
from pathlib import Path
from threading import Barrier, Lock

from scripts.server import run_v2_review_controls as runner


def test_parallel_probes_wait_for_cache_and_use_distinct_gpus(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    graph = tmp_path / "main/server_workspace/data/processed/hprc_r2_sv/full_segments.csv.gz"
    graph.parent.mkdir(parents=True)
    graph.write_bytes(b"synthetic graph")
    (tmp_path / "configs/entex_v1.json").write_text(json.dumps(dict(
        full_segments_sha256=hashlib.sha256(graph.read_bytes()).hexdigest())))
    (tmp_path / "configs/server_full_multicohort_20260806.json").write_text(json.dumps(dict(
        rotating_chromosome_folds=[dict(name="fold_a", test=["chr1"], validation=["chr2"]) ])))
    old = tmp_path / "main/server_workspace/results/full_multicohort_server_20260806/rotating_folds/hprc_r2/fold_a/seed_42/1hop/run_001/ckpt_1hop__toy.pt"
    old.parent.mkdir(parents=True)
    old.touch()
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.touch()
    monkeypatch.setattr("sys.argv", ["runner", "--main-checkout", str(tmp_path / "main"),
        "--out-root", str(tmp_path / "out"), "--contexts", "1hop", "--validation-only",
        "--candidate-checkpoint", f"v2={checkpoint}", "--candidate-checkpoint", f"v2_random={checkpoint}",
        "--models", "v2", "v2_random", "--probe-gpus", "1", "3",
        "--extraction-candidate-policy", "manuscript"])
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: "test-commit")
    cache_ready, active, seen = [], set(), []
    barrier, lock = Barrier(2), Lock()

    def fake_run(command, *, env, stdout, stderr, check):
        if command[1] == "-m":
            cache_ready.append(True)
            return
        assert cache_ready and "--validation-only" in command
        assert command[command.index("--extraction-candidate-policy") + 1] == "manuscript"
        gpu = env["CUDA_VISIBLE_DEVICES"]
        with lock:
            assert gpu not in active
            active.add(gpu)
            seen.append(gpu)
        barrier.wait(timeout=10)
        with lock:
            active.remove(gpu)
        stdout.write("synthetic execution test only\n")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.main() == 0
    receipt = json.loads((tmp_path / "out/status.json").read_text())
    assert receipt["status"] == "complete" and receipt["completed_commands"] == 5
    assert receipt["completed_command_indices"] == list(range(5))
    assert set(seen) == {"1", "3"} and len(seen) == 4
    assert len(list(Path("out").glob("command_*.log"))) == 5

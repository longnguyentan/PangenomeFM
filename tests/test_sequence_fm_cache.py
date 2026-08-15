from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.server import merge_node_sequence_fm_caches as merge
from scripts.server.prepare_node_sequence_fm_cache import balanced_sequence, sha256_file


def test_balanced_sequence_keeps_both_ends() -> None:
    sampled, changed = balanced_sequence("AAAACCCCGGGGTTTT", 9)
    assert changed
    assert sampled == "AAAANTTTT"
    assert len(sampled) == 9
    unchanged, changed = balanced_sequence("ACGT", 9)
    assert unchanged == "ACGT"
    assert not changed


def test_merge_sequence_shards_requires_exact_target_union(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "target.npz"
    np.savez(target, segid=np.array([1, 2, 3]))
    shards = []
    for index, segids in enumerate(([1, 3], [2])):
        path = tmp_path / f"shard_{index}.npz"
        np.savez(
            path,
            segid=np.array(segids),
            embeddings=np.full((len(segids), 4), index + 1, dtype=np.float32),
        )
        Path(f"{path}.audit.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "output_sha256": sha256_file(path),
                    "model_name": "toy-model",
                    "resolved_revision": "abc123",
                }
            )
        )
        shards.append(path)
    output = tmp_path / "merged.npz"
    monkeypatch.setattr(
        "sys.argv",
        [
            "merge_node_sequence_fm_caches.py",
            "--shard",
            *(str(path) for path in shards),
            "--target-cache",
            str(target),
            "--output",
            str(output),
        ],
    )
    assert merge.main() == 0
    with np.load(output) as cache:
        np.testing.assert_array_equal(cache["segid"], [1, 2, 3])
        assert cache["embeddings"].shape == (3, 4)
    audit = json.loads(Path(f"{output}.audit.json").read_text())
    assert audit["coverage_fraction"] == 1.0
    assert audit["downstream_label_access"] == "none"

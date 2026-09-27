from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.server import merge_node_sequence_fm_caches as merge
from scripts.server.complete_node_sequence_fm_cache import graph_targets, benchmark_targets, completion_commands
from scripts.server.prepare_node_sequence_fm_cache import (
    balanced_sequence,
    iter_segment_rows,
    sha256_file,
)


def test_balanced_sequence_keeps_both_ends() -> None:
    sampled, changed = balanced_sequence("AAAACCCCGGGGTTTT", 9)
    assert changed
    assert sampled == "AAAANTTTT"
    assert len(sampled) == 9
    unchanged, changed = balanced_sequence("ACGT", 9)
    assert unchanged == "ACGT"
    assert not changed


def test_segment_reader_accepts_sequence_larger_than_default_csv_limit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "full_segments.csv.gz"
    sequence = "A" * 200_000
    assert len(sequence) > 131_072
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "seq"])
        writer.writeheader()
        writer.writerow({"name": "s1", "seq": sequence})

    previous_limit = csv.field_size_limit(131_072)
    try:
        rows = list(iter_segment_rows(path))
    finally:
        csv.field_size_limit(previous_limit)

    assert len(rows) == 1
    assert rows[0]["name"] == "s1"
    assert rows[0]["seq"] == sequence


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
                    "full_segments_sha256": "same-canonical-graph",
                    "pooling": "mean", "maximum_token_length": 1000,
                    "maximum_raw_bases": 6000, "raw_sequence_sampling": "both ends",
                    "truncation_policy": "tokenizer", "fine_tuned": False,
                    "model_parameters_frozen": True, "downstream_label_access": "none",
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
    contract = merge.sequence_contract(output)
    assert contract["maximum_token_length"] == 1000
    # Legacy merged receipts contain model identity and source-shard hashes only.
    for key in merge.CONTRACT_FIELDS:
        if key not in ["model_name", "resolved_revision"]:
            del audit[key]
    Path(f"{output}.audit.json").write_text(json.dumps(audit))
    assert merge.sequence_contract(output) == contract
    source = Path(f"{shards[0]}.audit.json")
    changed = json.loads(source.read_text())
    changed["maximum_token_length"] = 500
    source.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="provenance changed"):
        merge.sequence_contract(output)


def test_completion_preserves_canonical_ids_and_refuses_missing_sequences(tmp_path):
    path = tmp_path / "segments.csv"
    path.write_text("name,seq\ns1,ACGT\ns2,A\ns3,C\n")
    ids, missing = graph_targets(path, np.array([0, 2]))
    np.testing.assert_array_equal(ids, [0, 1, 2])
    np.testing.assert_array_equal(missing, [1])
    with pytest.raises(ValueError, match="outside"):
        graph_targets(path, np.array([10]))
    path.write_text("name,seq\ns1,ACGT\ns2,*\n")
    with pytest.raises(ValueError, match="no sequence"):
        graph_targets(path, np.array([0]))
    path.write_text("name,seq\ns2,ACGT\n")
    with pytest.raises(ValueError, match="row indices"):
        graph_targets(path, np.array([]))


def test_benchmark_completion_uses_native_link_union_without_unrelated_nodes(tmp_path):
    graph = tmp_path / "segments.csv"
    graph.write_text("name,seq\ns1,A\ns2,G\ns3,C\ns4,*\n")
    links = tmp_path / "links.csv"
    links.write_text("from_seg,to_seg\ns1,s3\n")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(f"closure,links_path\n1hop,{links}\n")
    needed, scope = benchmark_targets(graph, manifest, "1hop")
    np.testing.assert_array_equal(needed, [0, 2])
    assert scope["graph_segments"] == 4 and scope["benchmark_segments"] == 2
    ids, missing = graph_targets(graph, np.array([0, 1]), needed)
    np.testing.assert_array_equal(ids, [0, 1, 2])
    np.testing.assert_array_equal(missing, [2])


def test_completion_shards_preserve_native_policy_and_merge_exact_targets(tmp_path):
    from argparse import Namespace
    args = Namespace(out_dir=tmp_path, manifest=tmp_path / "manifest.csv", shard_gpus=[0, 2, 3],
                     full_segments=tmp_path / "segments.csv", existing_cache=tmp_path / "original.npz",
                     batch_size=32, device="cuda")
    contract = dict(model_name="original-model", resolved_revision="pinned-revision",
                    maximum_token_length=1000, maximum_raw_bases=6000)
    commands, output = completion_commands(args, contract, 176052)
    assert len(commands) == 4 and output.name == "benchmark_nt.npz"
    for index, command in enumerate(commands[:-1]):
        for key, value in [("--shard-index", str(index)), ("--num-shards", "3"),
                           ("--revision", "pinned-revision"), ("--max-length", "1000"), ("--max-bases", "6000")]:
            assert command[command.index(key) + 1] == value
        assert str(tmp_path / f"missing_nt_{index}.npz") in commands[-1]
    assert str(args.existing_cache) in commands[-1]
    assert str(tmp_path / "all_targets.npz") in commands[-1]


def test_merge_rejects_same_model_with_different_preprocessing(tmp_path, monkeypatch):
    target = tmp_path / "target.npz"
    np.savez(target, segid=[0, 1])
    shards = []
    for i in range(2):
        path = tmp_path / f"s{i}.npz"
        np.savez(path, segid=[i], embeddings=np.zeros((1, 4), dtype=np.float32))
        audit = dict(status="complete", output_sha256=sha256_file(path), model_name="same-model",
                     resolved_revision="same-revision", full_segments_sha256="same-graph",
                     pooling="mean", maximum_token_length=1000 if i == 0 else 500,
                     maximum_raw_bases=6000, raw_sequence_sampling="both ends", truncation_policy="tokenizer",
                     fine_tuned=False, model_parameters_frozen=True, downstream_label_access="none")
        Path(f"{path}.audit.json").write_text(json.dumps(audit))
        shards.append(path)
    monkeypatch.setattr("sys.argv", ["merge", "--shard", *map(str, shards), "--target-cache", str(target),
                                     "--output", str(tmp_path / "merged.npz")])
    with pytest.raises(ValueError, match="preprocessing contracts differ"):
        merge.main()

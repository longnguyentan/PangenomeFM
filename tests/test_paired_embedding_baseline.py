from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.paired_embedding_baseline import pair_features, run


def test_pair_features_are_order_invariant() -> None:
    rng = np.random.default_rng(4)
    left = rng.normal(size=(5, 3))
    right = rng.normal(size=(5, 3))
    assert np.array_equal(pair_features(left, right), pair_features(right, left))


def test_frozen_embedding_adapter_preserves_example_ids(tmp_path) -> None:
    ids = np.array([f"n{i}" for i in range(12)])
    embeddings = np.arange(48, dtype=np.float32).reshape(12, 4)
    embedding_path = tmp_path / "embeddings.npz"
    np.savez_compressed(embedding_path, ids=ids, embeddings=embeddings)
    rows = []
    splits = ["train"] * 8 + ["validation"] * 4 + ["test"] * 4
    for index, split in enumerate(splits):
        rows.append(
            {
                "example_id": f"e{index}",
                "source_segment_name": f"n{index % 6}",
                "destination_segment_name": f"n{6 + index % 6}",
                "label": index % 2,
                "candidate_partition": split,
                "context": "strict",
                "chromosome": "chr22",
            }
        )
    manifest = pd.DataFrame(rows)
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    output = tmp_path / "result"
    summary = run(
        manifest_path=manifest_path,
        embeddings_path=embedding_path,
        out_dir=output,
        method_name="test-sequence-fm",
        require_complete=True,
    )
    predictions = pd.read_csv(output / "predictions.csv.gz")
    assert summary["converted_examples"] == len(manifest)
    assert set(predictions["example_id"]) == set(manifest["example_id"])
    assert summary["comparison_tier"].startswith("approximately_matched")


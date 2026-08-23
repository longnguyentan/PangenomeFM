from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.build_multimodal_pilot_npz import (
    build_branches,
    build_paths,
    masked_tokens,
)


def test_masked_tokens_never_masks_padding_and_preserves_labels() -> None:
    tokens = np.array([[1, 2, 3, 0], [4, 5, 1, 0]], dtype=np.int64)
    masked, labels = masked_tokens(tokens, np.array([0, 2]), 0.5, 7)
    assert np.array_equal(masked[:, -1], np.zeros(2, dtype=np.int64))
    selected = labels != -100
    assert selected.any()
    assert np.array_equal(labels[selected], tokens[selected])


def test_build_paths_creates_forward_and_swapped_examples() -> None:
    table = pd.DataFrame(
        {
            "path_id": ["p1"] * 6,
            "node_id": [f"n{i}" for i in range(6)],
            "position": list(range(6)),
            "split": ["test"] * 6,
        }
    )
    arrays, failures = build_paths(
        table,
        lookup={f"n{i}": i for i in range(6)},
        path_id_column="path_id",
        node_id_column="node_id",
        position_column="position",
        split_column="split",
        max_path_length=4,
        min_subpath_nodes=2,
    )
    assert failures == []
    assert arrays["path_nodes"].shape == (2, 3)
    assert arrays["path_order_labels"].tolist() == [1.0, 0.0]
    assert arrays["path_order_split"].tolist() == [2, 2]


def test_build_branches_keeps_one_positive_at_index_zero() -> None:
    table = pd.DataFrame(
        {
            "group_id": ["g"] * 4,
            "u_oid": ["1"] * 4,
            "candidate_v_oid": ["2", "3", "4", "5"],
            "label": [0, 1, 0, 0],
            "donor_split": ["validation"] * 4,
        }
    )
    arrays, failures = build_branches(
        table,
        lookup={str(i): i for i in range(6)},
        group_column="group_id",
        source_column="u_oid",
        candidate_column="candidate_v_oid",
        label_column="label",
        split_column="donor_split",
        choices=3,
    )
    assert failures == []
    assert arrays["branch_candidates"].tolist()[0][0] == 3
    assert arrays["branch_labels"].tolist() == [0]
    assert arrays["branch_split"].tolist() == [1]


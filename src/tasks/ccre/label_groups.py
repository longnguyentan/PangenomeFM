"""Shared cCRE label grouping utilities.

The original ENCODE cCRE mapping has 9 classes including background.  These
helpers define the reduced label spaces requested for downstream validation so
logistic, MLP, GAT, and embedding baselines all use identical class mappings.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from tasks.ccre.encoding import CCRE_CLASSES, CCRE_CLASS_TO_IDX


GROUP3_CLASSES = ["background", "enhancer_like", "other_ccre"]
GROUP4_CLASSES = [
    "background",
    "enhancer_like",
    "promoter_like",
    "tf_ctcf_or_open_chromatin",
]
GROUP5_CLASSES = [
    "background",
    "enhancer_like",
    "promoter_like",
    "tf_ctcf_associated",
    "open_chromatin",
]
BINARY_CLASSES = ["background", "ccre"]

_ENHANCER = {"pELS", "dELS"}
_PROMOTER = {"PLS", "CA-H3K4me3"}
_TF_CTCF = {"TF", "CA-CTCF", "CA-TF"}
_OPEN = {"CA"}

CATEGORY_GROUPS = {
    "enhancer_like": _ENHANCER,
    "promoter_like": _PROMOTER,
    "tf_ctcf_associated": _TF_CTCF,
    "open_chromatin": _OPEN,
    "dels": {"dELS"},
    "pels": {"pELS"},
    "pls": {"PLS"},
    "ctcf_tf": _TF_CTCF,
    "ca": {"CA"},
}


def canonical_scheme(scheme: str) -> str:
    scheme = scheme.lower().replace("-", "_")
    aliases = {
        "multiclass": "full9",
        "full": "full9",
        "9class": "full9",
        "9_class": "full9",
        "binary_ccre": "binary",
        "ccre_binary": "binary",
        "three": "group3",
        "3class": "group3",
        "3_class": "group3",
        "four": "group4",
        "4class": "group4",
        "4_class": "group4",
        "five": "group5",
        "5class": "group5",
        "5_class": "group5",
    }
    return aliases.get(scheme, scheme)


def class_names_for_scheme(scheme: str) -> list[str]:
    scheme = canonical_scheme(scheme)
    if scheme == "full9":
        return list(CCRE_CLASSES)
    if scheme == "binary":
        return list(BINARY_CLASSES)
    if scheme == "group3":
        return list(GROUP3_CLASSES)
    if scheme == "group4":
        return list(GROUP4_CLASSES)
    if scheme == "group5":
        return list(GROUP5_CLASSES)
    raise ValueError(f"Unknown cCRE label scheme: {scheme}")


def n_classes_for_scheme(scheme: str) -> int:
    return len(class_names_for_scheme(scheme))


def _class_name(label_idx: int) -> str:
    return CCRE_CLASSES[int(label_idx)]


def map_label_indices(
    labels: np.ndarray,
    *,
    scheme: str,
    ignore_index: int = -100,
) -> np.ndarray:
    """Map canonical 9-class integer labels into a reduced label space."""
    scheme = canonical_scheme(scheme)
    labels = np.asarray(labels)
    out = np.full(labels.shape, ignore_index, dtype=np.int64)
    valid = labels != ignore_index
    if scheme == "full9":
        out[valid] = labels[valid].astype(np.int64)
        return out
    if scheme == "binary":
        bg = CCRE_CLASS_TO_IDX["background"]
        out[valid] = (labels[valid] != bg).astype(np.int64)
        return out

    for idx in np.where(valid)[0]:
        cls = _class_name(int(labels[idx]))
        if cls == "background":
            out[idx] = 0
        elif cls in _ENHANCER:
            out[idx] = 1
        elif scheme == "group3":
            out[idx] = 2
        elif cls in _PROMOTER:
            out[idx] = 2
        elif scheme == "group4":
            out[idx] = 3
        elif cls in _TF_CTCF:
            out[idx] = 3
        elif cls in _OPEN:
            out[idx] = 4
        else:
            raise ValueError(f"Unhandled cCRE class for {scheme}: {cls}")
    return out


def category_binary_labels(
    labels: np.ndarray,
    *,
    positive_group: str,
    ignore_index: int = -100,
    background_only_negative: bool = True,
) -> np.ndarray:
    """Build a one-vs-background category-specific binary task.

    Positive nodes belong to ``positive_group``.  By default, negatives are
    only background nodes and all other cCRE classes are ignored.  This matches
    the biologically interpretable "category vs background" tasks Prof asked
    for.
    """
    key = positive_group.lower().replace("-", "_")
    if key not in CATEGORY_GROUPS:
        raise ValueError(
            f"Unknown positive_group={positive_group!r}. "
            f"Choose one of: {sorted(CATEGORY_GROUPS)}"
        )
    positive = CATEGORY_GROUPS[key]
    labels = np.asarray(labels)
    out = np.full(labels.shape, ignore_index, dtype=np.int64)
    for idx in np.where(labels != ignore_index)[0]:
        cls = _class_name(int(labels[idx]))
        if cls in positive:
            out[idx] = 1
        elif cls == "background" or not background_only_negative:
            out[idx] = 0
    return out


def describe_scheme(scheme: str, positive_group: str | None = None) -> dict[str, object]:
    scheme = canonical_scheme(scheme)
    if scheme == "category_binary":
        if positive_group is None:
            raise ValueError("positive_group is required for category_binary")
        key = positive_group.lower().replace("-", "_")
        return {
            "scheme": scheme,
            "positive_group": key,
            "classes": ["background", positive_group],
            "positive_original_labels": sorted(CATEGORY_GROUPS[key]),
        }
    return {"scheme": scheme, "classes": class_names_for_scheme(scheme)}


def count_labels(labels: Iterable[int], class_names: list[str]) -> dict[str, int]:
    arr = np.asarray(list(labels), dtype=np.int64)
    return {name: int((arr == i).sum()) for i, name in enumerate(class_names)}

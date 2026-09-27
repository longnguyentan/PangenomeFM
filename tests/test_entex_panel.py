import pytest

from tasks.entex.panel import partition_counts


FOLDS = [dict(name=f"fold_{i}", test=[f"chr{i}"], validation=[f"chr{i % 3 + 1}"])
         for i in range(1, 4)]


def test_all_partitions_have_measured_both_class_support():
    counts = {f"chr{i}": {"positive": i * 100, "negative": 2000} for i in range(1, 4)}
    out = partition_counts(counts, FOLDS, 100)
    assert len(out) == 9
    assert out.groupby("fold").positive.sum().eq(600).all()
    assert set(out.partition) == {"train", "validation", "test"}


def test_missing_or_unpowered_chromosome_group_is_rejected():
    with pytest.raises(ValueError, match="requires 100"):
        partition_counts({"chr1": {"positive": 500, "negative": 500}}, FOLDS, 100)
    with pytest.raises(ValueError, match="outside manuscript"):
        partition_counts({"chr99": {"positive": 500, "negative": 500}}, FOLDS, 100)


def test_overlap_in_fold_partitions_is_rejected():
    folds = [dict(name="bad", test=["chr1"], validation=["chr1"])]
    with pytest.raises(ValueError, match="Invalid chromosome"):
        partition_counts({"chr1": {"positive": 500, "negative": 500}}, folds, 100)

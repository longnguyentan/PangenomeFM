import numpy as np
import pytest

from scripts.server.complete_reference_topology_cache import append_only


def test_cache_extension_preserves_native_values_and_rejects_overwrite():
    ids = np.array([4, 1])
    values = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    out_ids, out_values = append_only(ids, values, {9: np.array([5., 6.])})
    assert np.array_equal(out_ids, [4, 1, 9])
    assert np.array_equal(out_values[:2], values)
    assert out_values.dtype == values.dtype
    with pytest.raises(ValueError, match='replace'):
        append_only(ids, values, {1: np.zeros(2)})
    with pytest.raises(ValueError, match='Invalid'):
        append_only(ids, values, {9: np.full(2, np.nan)})

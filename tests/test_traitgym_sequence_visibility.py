import numpy as np
import pytest

from scripts.server.audit_traitgym_sequence_visibility import raw_base_retained
from scripts.server.prepare_node_sequence_fm_cache import balanced_sequence


def test_visibility_matches_native_balanced_sampling_at_boundaries():
    for length in [5, 6, 7, 20]:
        sequence = "ABCDEFGHIJKLMNOPQRST"[:length]
        sampled, _ = balanced_sequence(sequence, 6)
        keep = raw_base_retained(np.arange(length), np.full(length, length), 6)
        # A unique source character identifies each true base; the inserted N
        # sentinel is not a retained reference base.
        original = sequence if length <= 6 else sampled[:2] + sampled[3:]
        assert "".join(c for c, retained in zip(sequence, keep) if retained) == original
    np.testing.assert_array_equal(raw_base_retained(np.array([2998, 2999, 6999, 7000]),
        np.full(4, 10000), 6000), [True, False, False, True])


def test_visibility_rejects_bad_mapping():
    for offset in [-1, 10, .5, np.nan]:
        with pytest.raises(ValueError, match="Invalid"):
            raw_base_retained(np.array([offset]), np.array([10]), 6)

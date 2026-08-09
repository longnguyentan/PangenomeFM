from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_scaling_benchmark.py"
SPEC = importlib.util.spec_from_file_location("run_scaling_benchmark", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_projection_separates_fixed_overhead_from_slice_slope() -> None:
    projection = MODULE.project_full_runtime(
        [4, 8, 16, 32],
        [11.0, 13.0, 17.0, 25.0],
        full_train_slices=100,
        projected_epochs=10,
    )

    assert projection["projection_method"] == "ols_intercept_plus_training_slice_slope"
    assert projection["fitted_fixed_seconds_per_epoch"] == pytest.approx(9.0)
    assert projection["fitted_seconds_per_training_slice"] == pytest.approx(0.5)
    assert projection["estimated_seconds_per_epoch_at_full_scale"] == pytest.approx(59.0)
    assert projection["linear_projection_seconds"] == pytest.approx(590.0)


def test_projection_has_explicit_single_probe_fallback() -> None:
    projection = MODULE.project_full_runtime(
        [20],
        [10.0],
        full_train_slices=100,
        projected_epochs=2,
    )

    assert projection["projection_method"] == "single_probe_proportional_fallback"
    assert projection["estimated_seconds_per_epoch_at_full_scale"] == pytest.approx(50.0)
    assert projection["linear_projection_seconds"] == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("counts", "seconds", "full_slices", "epochs"),
    [([], [], 10, 1), ([1], [], 10, 1), ([0], [1.0], 10, 1), ([1], [0.0], 10, 1), ([1], [1.0], 0, 1)],
)
def test_projection_rejects_invalid_inputs(
    counts: list[int],
    seconds: list[float],
    full_slices: int,
    epochs: int,
) -> None:
    with pytest.raises(ValueError):
        MODULE.project_full_runtime(
            counts,
            seconds,
            full_train_slices=full_slices,
            projected_epochs=epochs,
        )

from __future__ import annotations

import numpy as np
import pytest

from envstretcher.mapping_definition import MappingStrategy
from main import run_synthetic_warp


@pytest.mark.parametrize(
    "triangles",
    [
        None,
        (
            ("1", "2", "3"),
            ("1", "3", "4"),
        ),
    ],
)
def test_synthetic_warp_system_scenario(
    triangles: tuple[tuple[str, str, str], ...] | None,
) -> None:
    _, target_grid, expected_values = run_synthetic_warp(
        triangles=triangles,
    )

    assert target_grid.values.shape == (6, 6)
    assert np.all(np.isfinite(target_grid.values))
    assert np.allclose(
        target_grid.values,
        expected_values,
        rtol=1e-12,
        atol=1e-12,
    )


def test_synthetic_warp_preconstructed_triangles_matches_delaunay() -> None:
    _, delaunay_target, _ = run_synthetic_warp(triangles=None)

    _, explicit_target, _ = run_synthetic_warp(
        triangles=(
            ("1", "2", "3"),
            ("1", "3", "4"),
        )
    )

    assert np.allclose(
        delaunay_target.values,
        explicit_target.values,
        rtol=1e-12,
        atol=1e-12,
    )


def test_synthetic_warp_supports_max_cell_area_strategy() -> None:
    _, target_grid, expected_values = run_synthetic_warp(
        strategy=MappingStrategy.MAX_CELL_AREA,
    )

    assert np.allclose(
        target_grid.values,
        expected_values,
        rtol=1e-12,
        atol=1e-12,
    )


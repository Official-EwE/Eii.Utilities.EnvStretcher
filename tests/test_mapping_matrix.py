from __future__ import annotations

import numpy as np
import pytest

from envstretcher.bilinear_sampler import BilinearSampler
from envstretcher.control_point import ControlPoint
from envstretcher.mapping_definition import MappingStrategy
from envstretcher.mapping_matrix import MappingMatrix
from envstretcher.piecewise_affine_transform import PiecewiseAffineTransform
from envstretcher.raster_geometry import RasterGeometry
from envstretcher.raster_grid import RasterGrid
from envstretcher.raster_warper import RasterWarper


def _create_source_grid(offset: float = 0.0) -> RasterGrid:
    geometry = RasterGeometry(
        rows=10,
        columns=10,
        x_min=0.0,
        y_min=0.0,
        cell_size=1.0,
    )

    values = np.empty((geometry.rows, geometry.columns), dtype=np.float64)

    for row in range(geometry.rows):
        for col in range(geometry.columns):
            x, y = geometry.cell_center(row, col)
            values[row, col] = 100.0 + 2.0 * x + 3.0 * y + offset

    return RasterGrid(geometry=geometry, values=values, nodata=np.nan)


def _create_transform() -> PiecewiseAffineTransform:
    points = [
        ControlPoint(id="1", source_x=0.0, source_y=0.0, target_x=0.0, target_y=0.0),
        ControlPoint(id="2", source_x=10.0, source_y=0.0, target_x=6.0, target_y=0.0),
        ControlPoint(id="3", source_x=10.0, source_y=10.0, target_x=6.0, target_y=6.0),
        ControlPoint(id="4", source_x=0.0, source_y=10.0, target_x=0.0, target_y=6.0),
    ]

    return PiecewiseAffineTransform(control_points=points)


def _create_target_geometry() -> RasterGeometry:
    return RasterGeometry(
        rows=6,
        columns=6,
        x_min=0.0,
        y_min=0.0,
        cell_size=1.0,
    )


def test_mapping_matrix_roundtrip_and_reuse(tmp_path) -> None:
    source_grid = _create_source_grid(offset=0.0)
    transform = _create_transform()
    target_geometry = _create_target_geometry()

    warper = RasterWarper(
        sampler=BilinearSampler(source_grid),
        transform=transform,
    )

    matrix = warper.build_mapping_matrix(
        target_geometry=target_geometry,
        strategy=MappingStrategy.MAX_CELL_AREA,
    )

    matrix_file = tmp_path / "mapping_matrix.npz"
    matrix.write(matrix_file)

    loaded = MappingMatrix.read(matrix_file)

    assert loaded.strategy == MappingStrategy.MAX_CELL_AREA
    assert np.array_equal(loaded.success, matrix.success)

    warped_a = warper.warp_with_mapping(loaded)

    source_grid_b = _create_source_grid(offset=7.0)
    warper_b = RasterWarper(
        sampler=BilinearSampler(source_grid_b),
        transform=transform,
    )

    warped_b = warper_b.warp_with_mapping(loaded)

    assert np.all(np.isfinite(warped_a.values))
    assert np.all(np.isfinite(warped_b.values))

    diff = warped_b.values - warped_a.values
    assert np.allclose(diff, 7.0, rtol=1e-12, atol=1e-12)


def test_mapping_matrix_rejects_source_geometry_mismatch() -> None:
    source_grid = _create_source_grid()
    transform = _create_transform()

    warper = RasterWarper(
        sampler=BilinearSampler(source_grid),
        transform=transform,
    )

    matrix = warper.build_mapping_matrix(
        target_geometry=_create_target_geometry(),
        strategy=MappingStrategy.CENTROID,
    )

    mismatched_geometry = RasterGeometry(
        rows=11,
        columns=10,
        x_min=0.0,
        y_min=0.0,
        cell_size=1.0,
    )
    mismatch_grid = RasterGrid.create_filled(
        geometry=mismatched_geometry,
        value=1.0,
        nodata=np.nan,
    )

    mismatch_warper = RasterWarper(
        sampler=BilinearSampler(mismatch_grid),
        transform=transform,
    )

    with pytest.raises(ValueError, match="Source geometry mismatch"):
        mismatch_warper.warp_with_mapping(matrix)


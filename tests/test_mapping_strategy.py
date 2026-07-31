from __future__ import annotations

import numpy as np

from envstretcher.bilinear_sampler import BilinearSampler
from envstretcher.control_point import ControlPoint
from envstretcher.mapping_definition import MappingStrategy
from envstretcher.piecewise_affine_transform import PiecewiseAffineTransform
from envstretcher.raster_geometry import RasterGeometry
from envstretcher.raster_grid import RasterGrid
from envstretcher.raster_warper import RasterWarper


def test_max_cell_area_maps_cell_when_centroid_is_outside_triangle() -> None:
    source_geometry = RasterGeometry(
        rows=4,
        columns=4,
        x_min=0.0,
        y_min=0.0,
        cell_size=1.0,
    )
    source_grid = RasterGrid.create_filled(
        geometry=source_geometry,
        value=42.0,
        nodata=np.nan,
    )

    target_geometry = RasterGeometry(
        rows=1,
        columns=1,
        x_min=0.0,
        y_min=0.0,
        cell_size=1.0,
    )

    # This narrow target triangle overlaps the cell but does not contain
    # the target-cell centroid (0.5, 0.5).
    control_points = [
        ControlPoint(
            id="1",
            source_x=1.0,
            source_y=1.0,
            target_x=0.0,
            target_y=0.0,
        ),
        ControlPoint(
            id="2",
            source_x=2.0,
            source_y=1.0,
            target_x=1.0,
            target_y=0.0,
        ),
        ControlPoint(
            id="3",
            source_x=1.0,
            source_y=1.2,
            target_x=0.0,
            target_y=0.2,
        ),
    ]

    transform = PiecewiseAffineTransform(
        control_points=control_points,
        triangles=(("1", "2", "3"),),
    )
    warper = RasterWarper(
        sampler=BilinearSampler(source_grid),
        transform=transform,
    )

    centroid_result = warper.warp(
        target_geometry=target_geometry,
        strategy=MappingStrategy.CENTROID,
    )
    max_area_result = warper.warp(
        target_geometry=target_geometry,
        strategy=MappingStrategy.MAX_CELL_AREA,
    )

    assert np.isnan(centroid_result.values[0, 0])
    assert max_area_result.values[0, 0] == 42.0


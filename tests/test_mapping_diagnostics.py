from __future__ import annotations

import numpy as np

from envstretcher.coverage_code import CoverageCode
from diagnostics.mapping_diagnostics import MappingDiagnostics
from envstretcher.bilinear_sampler import BilinearSampler
from envstretcher.control_point import ControlPoint
from envstretcher.piecewise_affine_transform import PiecewiseAffineTransform
from envstretcher.raster_geometry import RasterGeometry
from envstretcher.raster_grid import RasterGrid

SOURCE_NODATA = -9999.0


def test_mapping_diagnostics_classifies_all_cell_types() -> None:
    """
    Verify the four diagnostic classifications using:

    - a fictional 10 x 10 source raster;
    - a fictional 6 x 6 EwE basemap;
    - a target mapping surface that deliberately excludes some water cells;
    - a source NoData cell that invalidates one bilinear sample.
    """

    source = create_source_raster()
    basemap = create_target_basemap()
    transform = create_transform()

    sampler = BilinearSampler(source)

    diagnostics = MappingDiagnostics(
        transform=transform,
        sampler=sampler,
    ).classify(basemap)

    expected = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 4, 2, 2, 2, 1],
            [1, 4, 3, 2, 2, 1],
            [1, 4, 2, 2, 2, 1],
            [1, 4, 4, 4, 4, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=np.int16,
    )

    np.testing.assert_array_equal(
        diagnostics.values,
        expected,
    )


def test_mapping_diagnostics_contains_every_coverage_code() -> None:
    """
    Provide a more descriptive failure if one diagnostic category disappears.
    """

    source = create_source_raster()
    basemap = create_target_basemap()
    transform = create_transform()

    diagnostics = MappingDiagnostics(
        transform=transform,
        sampler=BilinearSampler(source),
    ).classify(basemap)

    actual_codes = set(
        int(value)
        for value in np.unique(diagnostics.values)
    )

    expected_codes = {
        int(CoverageCode.LAND),
        int(CoverageCode.VALID_DATA),
        int(CoverageCode.SOURCE_NODATA),
        int(CoverageCode.OUTSIDE_MAPPING),
    }

    assert actual_codes == expected_codes


def create_source_raster() -> RasterGrid:
    """
    Create a 10 x 10 source raster containing a simple linear field.

    The source extent is:

        x: 0 .. 10
        y: 0 .. 10

    Cell values follow:

        value = 100 + 2x + 3y

    One source cell is changed to NoData so that the target cell at
    row 2, column 2 produces SOURCE_NODATA.
    """

    geometry = RasterGeometry(
        rows=10,
        columns=10,
        x_min=0.0,
        y_min=0.0,
        cell_size=1.0,
    )

    values = np.empty(
        shape=(geometry.rows, geometry.columns),
    )

    for row in range(geometry.rows):
        for column in range(geometry.columns):
            x, y = geometry.cell_center(row, column)
            values[row, column] = 100.0 + 2.0 * x + 3.0 * y

    # Target cell centre (2.5, 3.5) maps to source coordinate (3, 5).
    #
    # That coordinate lies between source rows 4/5 and columns 2/3.
    # Setting one contributing neighbour to NoData therefore makes
    # the bilinear sample invalid.
    values[4, 2] = SOURCE_NODATA

    return RasterGrid(
        geometry=geometry,
        values=values,
        nodata=SOURCE_NODATA,
    )


def create_target_basemap() -> RasterGrid:
    """
    Create a 6 x 6 fictional EwE basemap.

    The outer ring is land, represented by depth 0.
    The inner 4 x 4 area is water, represented by depth 10.
    """

    geometry = RasterGeometry(
        rows=6,
        columns=6,
        x_min=0.0,
        y_min=0.0,
        cell_size=1.0,
    )

    values = np.array(
        [
            [0,  0,  0,  0,  0, 0],
            [0, 10, 10, 10, 10, 0],
            [0, 10, 10, 10, 10, 0],
            [0, 10, 10, 10, 10, 0],
            [0, 10, 10, 10, 10, 0],
            [0,  0,  0,  0,  0, 0],
        ],
        dtype=float,
    )

    return RasterGrid(
        geometry=geometry,
        values=values,
        nodata=-9999.0,
    )


def create_transform() -> PiecewiseAffineTransform:
    """
    Map a target rectangle from (2, 2) .. (5, 5) onto a source
    rectangle from (2, 2) .. (8, 8).

    Some EwE water-cell centres lie outside this target rectangle,
    deliberately producing OUTSIDE_MAPPING diagnostics.
    """

    control_points = [
        ControlPoint(
            id="1",
            target_x=2.0,
            target_y=2.0,
            source_x=2.0,
            source_y=2.0,
        ),
        ControlPoint(
            id="2",
            target_x=5.0,
            target_y=2.0,
            source_x=8.0,
            source_y=2.0,
        ),
        ControlPoint(
            id="3",
            target_x=5.0,
            target_y=5.0,
            source_x=8.0,
            source_y=8.0,
        ),
        ControlPoint(
            id="4",
            target_x=2.0,
            target_y=5.0,
            source_x=2.0,
            source_y=8.0,
        ),
    ]

    return PiecewiseAffineTransform(control_points)
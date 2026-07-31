from __future__ import annotations

import numpy as np

from diagnostics.mapping_diagnostics import MappingDiagnostics
from io.ascii_reader import AsciiGridReader
from io.ascii_writer import AsciiGridWriter
from envstretcher.batch_ascii_mapper import BatchAsciiMapper
from envstretcher.bilinear_sampler import BilinearSampler
from envstretcher.control_point import ControlPoint
from envstretcher.coverage_code import CoverageCode
from envstretcher.mapping_definition import MappingStrategy
from envstretcher.piecewise_affine_transform import PiecewiseAffineTransform
from envstretcher.raster_geometry import RasterGeometry
from envstretcher.raster_grid import RasterGrid


def _create_source_grid(offset: float) -> RasterGrid:
    geometry = RasterGeometry(
        rows=10,
        columns=10,
        x_min=0.0,
        y_min=0.0,
        cell_size=1.0,
    )
    values = np.empty((10, 10), dtype=np.float64)

    for row in range(10):
        for col in range(10):
            x, y = geometry.cell_center(row, col)
            values[row, col] = 100.0 + 2.0 * x + 3.0 * y + offset

    return RasterGrid(
        geometry=geometry,
        values=values,
        nodata=-9999.0,
    )


def _create_basemap() -> RasterGrid:
    geometry = RasterGeometry(
        rows=6,
        columns=6,
        x_min=0.0,
        y_min=0.0,
        cell_size=1.0,
    )

    values = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 10, 10, 10, 10, 0],
            [0, 10, 10, 10, 10, 0],
            [0, 10, 10, 10, 10, 0],
            [0, 10, 10, 10, 10, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=float,
    )

    return RasterGrid(
        geometry=geometry,
        values=values,
        nodata=-9999.0,
    )


def _create_transform() -> PiecewiseAffineTransform:
    control_points = [
        ControlPoint(id="1", source_x=0.0, source_y=0.0, target_x=0.0, target_y=0.0),
        ControlPoint(id="2", source_x=10.0, source_y=0.0, target_x=6.0, target_y=0.0),
        ControlPoint(id="3", source_x=10.0, source_y=10.0, target_x=6.0, target_y=6.0),
        ControlPoint(id="4", source_x=0.0, source_y=10.0, target_x=0.0, target_y=6.0),
    ]

    return PiecewiseAffineTransform(control_points)


def _create_target_geometry() -> RasterGeometry:
    return RasterGeometry(
        rows=6,
        columns=6,
        x_min=0.0,
        y_min=0.0,
        cell_size=1.0,
    )


def test_mapping_diagnostics_classify_warped(tmp_path) -> None:
    source_file = tmp_path / "source.asc"
    source_grid = _create_source_grid(offset=0.0)
    basemap = _create_basemap()
    transform = _create_transform()

    AsciiGridWriter.write(source_grid, source_file)

    warper = BatchAsciiMapper(
        transform=transform,
        target_geometry=_create_target_geometry(),
        strategy=MappingStrategy.MAX_CELL_AREA,
    )

    matrix = warper.build_mapping_matrix(
        template_source_ascii=source_file,
    )

    source_grid_loaded = AsciiGridReader.read(source_file)
    warped = matrix.warp(
        sampler=BilinearSampler(source_grid_loaded),
    )

    diagnostics_gen = MappingDiagnostics(
        transform=transform,
        sampler=BilinearSampler(source_grid_loaded),
    )

    diag_grid = diagnostics_gen.classify_warped(
        warped=warped,
        basemap=basemap,
    )

    assert diag_grid.values.shape == (6, 6)

    land_count = np.count_nonzero(diag_grid.values == CoverageCode.LAND)
    valid_count = np.count_nonzero(diag_grid.values == CoverageCode.VALID_DATA)

    assert land_count > 0
    assert valid_count > 0
    assert land_count + valid_count == 36


def test_batch_warp_with_diagnostics(tmp_path) -> None:
    source_file = tmp_path / "source.asc"
    basemap_file = tmp_path / "basemap.asc"

    source_grid = _create_source_grid(offset=0.0)
    basemap = _create_basemap()

    AsciiGridWriter.write(source_grid, source_file)
    AsciiGridWriter.write(basemap, basemap_file)

    mapper = BatchAsciiMapper(
        transform=_create_transform(),
        target_geometry=_create_target_geometry(),
        strategy=MappingStrategy.MAX_CELL_AREA,
    )

    matrix_file = tmp_path / "mapping.npz"
    matrix = mapper.build_mapping_matrix(
        template_source_ascii=source_file,
        output_matrix_file=matrix_file,
    )

    output_dir = tmp_path / "out"
    diag_dir = tmp_path / "diag"

    stats = mapper.warp_files_with_diagnostics(
        source_files=[source_file],
        output_directory=output_dir,
        matrix=matrix,
        basemap=basemap,
        diagnostics_directory=diag_dir,
    )

    assert stats.processed_files == 1

    warped = AsciiGridReader.read(output_dir / "source.asc")
    diag = AsciiGridReader.read(diag_dir / "source_diag.asc")

    assert warped.values.shape == (6, 6)
    assert diag.values.shape == (6, 6)
    assert np.all(np.isfinite(warped.values[1:5, 1:5]))




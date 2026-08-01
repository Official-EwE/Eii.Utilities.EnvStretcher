from __future__ import annotations

import numpy as np

from envstretcher.io.ascii_reader import AsciiGridReader
from envstretcher.io.ascii_writer import AsciiGridWriter
from envstretcher.batch_ascii_mapper import BatchAsciiMapper
from envstretcher.bilinear_sampler import BilinearSampler
from envstretcher.control_point import ControlPoint
from envstretcher.mapping_definition import MappingStrategy
from envstretcher.mapping_matrix import MappingMatrix
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


def _create_basemap() -> RasterGrid:
    geometry = _create_target_geometry()
    values = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 10, 10, 10, 10, 0],
            [0, 10, 10, 10, 10, 0],
            [0, 10, 10, 10, 10, 0],
            [0, 10, 10, 10, 10, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )

    return RasterGrid(
        geometry=geometry,
        values=values,
        nodata=-9999.0,
    )


def _create_index_pattern_grid(
    geometry: RasterGeometry,
    offset: float,
) -> RasterGrid:
    values = np.empty(
        (geometry.rows, geometry.columns),
        dtype=np.float64,
    )

    for row in range(geometry.rows):
        for col in range(geometry.columns):
            values[row, col] = row * 100.0 + col + offset

    return RasterGrid(
        geometry=geometry,
        values=values,
        nodata=-9999.0,
    )


def test_batch_ascii_mapper_two_phase_workflow(tmp_path) -> None:
    source_a = tmp_path / "month_01.asc"
    source_b = tmp_path / "month_02.asc"

    AsciiGridWriter.write(_create_source_grid(offset=0.0), source_a)
    AsciiGridWriter.write(_create_source_grid(offset=12.0), source_b)

    mapper = BatchAsciiMapper(
        transform=_create_transform(),
        target_geometry=_create_target_geometry(),
        strategy=MappingStrategy.MAX_CELL_AREA,
    )

    matrix_file = tmp_path / "gol_to_ewe_mapping.npz"
    matrix = mapper.build_mapping_matrix(
        template_source_ascii=source_a,
        output_matrix_file=matrix_file,
    )

    loaded_matrix = MappingMatrix.read(matrix_file)
    output_dir = tmp_path / "out"

    stats = mapper.warp_glob(
        source_directory=tmp_path,
        pattern="month_*.asc",
        output_directory=output_dir,
        matrix=loaded_matrix,
    )

    assert stats.processed_files == 2

    out_a = AsciiGridReader.read(output_dir / "month_01.asc")
    out_b = AsciiGridReader.read(output_dir / "month_02.asc")

    assert out_a.values.shape == (6, 6)
    assert out_b.values.shape == (6, 6)

    assert np.all(np.isfinite(out_a.values))
    assert np.all(np.isfinite(out_b.values))

    assert np.allclose(
        out_b.values - out_a.values,
        12.0,
        rtol=1e-6,
        atol=1e-6,
    )


def test_batch_ascii_mapper_warp_glob_recurses_and_preserves_structure(
    tmp_path,
) -> None:
    source_root = tmp_path / "input"
    source_a = source_root / "year_2000" / "month.asc"
    source_b = source_root / "year_2001" / "month.asc"

    AsciiGridWriter.write(_create_source_grid(offset=0.0), source_a)
    AsciiGridWriter.write(_create_source_grid(offset=7.0), source_b)

    mapper = BatchAsciiMapper(
        transform=_create_transform(),
        target_geometry=_create_target_geometry(),
        strategy=MappingStrategy.MAX_CELL_AREA,
    )

    matrix_file = tmp_path / "mapping.npz"
    mapper.build_mapping_matrix(
        template_source_ascii=source_a,
        output_matrix_file=matrix_file,
    )
    matrix = MappingMatrix.read(matrix_file)

    output_dir = tmp_path / "out"
    stats = mapper.warp_glob(
        source_directory=source_root,
        pattern="*.asc",
        output_directory=output_dir,
        matrix=matrix,
    )

    assert stats.processed_files == 2

    out_a = AsciiGridReader.read(output_dir / "year_2000" / "month.asc")
    out_b = AsciiGridReader.read(output_dir / "year_2001" / "month.asc")

    assert out_a.values.shape == (6, 6)
    assert out_b.values.shape == (6, 6)
    assert np.allclose(
        out_b.values - out_a.values,
        7.0,
        rtol=1e-6,
        atol=1e-6,
    )


def test_batch_ascii_mapper_phase1_coverage_diagnostics(tmp_path) -> None:
    source_file = tmp_path / "source.asc"
    basemap = _create_basemap()

    AsciiGridWriter.write(_create_source_grid(offset=0.0), source_file)

    mapper = BatchAsciiMapper(
        transform=_create_transform(),
        target_geometry=_create_target_geometry(),
        strategy=MappingStrategy.MAX_CELL_AREA,
    )

    diagnostics_file = tmp_path / "phase1_diag.asc"
    summary = mapper.diagnose_template_coverage(
        template_source_ascii=source_file,
        basemap=basemap,
        diagnostics_output_file=diagnostics_file,
    )

    diagnostics_grid = AsciiGridReader.read(diagnostics_file)

    assert diagnostics_grid.values.shape == (6, 6)
    assert summary.water_cells == 16
    assert summary.mapped_cells == 16
    assert summary.outside_mapping_cells == 0
    assert summary.source_nodata_cells == 0
    assert summary.all_water_cells_covered


def test_batch_ascii_mapper_index_control_points_mode(tmp_path) -> None:
    source_geometry = RasterGeometry(
        rows=10,
        columns=10,
        x_min=100.0,
        y_min=200.0,
        cell_size=2.0,
    )
    source_values = np.ones((10, 10), dtype=np.float64)
    source_grid = RasterGrid(
        geometry=source_geometry,
        values=source_values,
        nodata=-9999.0,
    )
    source_file = tmp_path / "source.asc"
    AsciiGridWriter.write(source_grid, source_file)

    target_geometry = RasterGeometry(
        rows=6,
        columns=6,
        x_min=-50.0,
        y_min=20.0,
        cell_size=5.0,
    )
    basemap = RasterGrid(
        geometry=target_geometry,
        values=np.ones((6, 6), dtype=np.float64),
        nodata=-9999.0,
    )

    index_transform = PiecewiseAffineTransform(
        [
            ControlPoint(id="1", source_x=0.0, source_y=0.0, target_x=0.0, target_y=0.0),
            ControlPoint(id="2", source_x=10.0, source_y=0.0, target_x=6.0, target_y=0.0),
            ControlPoint(id="3", source_x=10.0, source_y=10.0, target_x=6.0, target_y=6.0),
            ControlPoint(id="4", source_x=0.0, source_y=10.0, target_x=0.0, target_y=6.0),
        ]
    )

    mapper_world = BatchAsciiMapper(
        transform=index_transform,
        target_geometry=target_geometry,
        strategy=MappingStrategy.CENTROID,
    )
    world_summary = mapper_world.diagnose_template_coverage(
        template_source_ascii=source_file,
        basemap=basemap,
    )
    assert world_summary.mapped_cells == 0

    mapper_index = BatchAsciiMapper(
        transform=index_transform,
        target_geometry=target_geometry,
        strategy=MappingStrategy.CENTROID,
        use_index_control_points=True,
    )
    index_summary = mapper_index.diagnose_template_coverage(
        template_source_ascii=source_file,
        basemap=basemap,
    )
    assert index_summary.mapped_cells == 36
    assert index_summary.all_water_cells_covered

    matrix = mapper_index.build_mapping_matrix(
        template_source_ascii=source_file,
    )
    assert int(matrix.success.sum()) == 36


def test_batch_ascii_mapper_index_mode_remaps_matrix_to_source_geometry(
    tmp_path,
) -> None:
    source_geometry_a = RasterGeometry(
        rows=10,
        columns=10,
        x_min=-1.0,
        y_min=37.0,
        cell_size=0.0833,
    )
    source_geometry_b = RasterGeometry(
        rows=10,
        columns=10,
        x_min=-1.041666666667,
        y_min=37.041665395101,
        cell_size=0.083333333333,
    )

    source_a = tmp_path / "source_a.asc"
    source_b = tmp_path / "source_b.asc"
    AsciiGridWriter.write(
        _create_index_pattern_grid(source_geometry_a, offset=0.0),
        source_a,
    )
    AsciiGridWriter.write(
        _create_index_pattern_grid(source_geometry_b, offset=9.0),
        source_b,
    )

    target_geometry = _create_target_geometry()
    index_transform = PiecewiseAffineTransform(
        [
            ControlPoint(id="1", source_x=0.0, source_y=0.0, target_x=0.0, target_y=0.0),
            ControlPoint(id="2", source_x=10.0, source_y=0.0, target_x=6.0, target_y=0.0),
            ControlPoint(id="3", source_x=10.0, source_y=10.0, target_x=6.0, target_y=6.0),
            ControlPoint(id="4", source_x=0.0, source_y=10.0, target_x=0.0, target_y=6.0),
        ]
    )

    mapper = BatchAsciiMapper(
        transform=index_transform,
        target_geometry=target_geometry,
        strategy=MappingStrategy.CENTROID,
        use_index_control_points=True,
    )

    matrix_from_a = mapper.build_mapping_matrix(
        template_source_ascii=source_a,
    )
    output_dir = tmp_path / "out"
    mapper.warp_files(
        source_files=[source_b],
        output_directory=output_dir,
        matrix=matrix_from_a,
    )

    warped_with_remap = AsciiGridReader.read(output_dir / "source_b.asc")

    matrix_from_b = mapper.build_mapping_matrix(
        template_source_ascii=source_b,
    )
    expected = matrix_from_b.warp(
        sampler=BilinearSampler(AsciiGridReader.read(source_b)),
    )

    assert np.allclose(
        warped_with_remap.values,
        expected.values,
        equal_nan=True,
    )

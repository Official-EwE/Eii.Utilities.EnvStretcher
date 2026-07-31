from __future__ import annotations

import numpy as np

from io.ascii_reader import AsciiGridReader
from io.ascii_writer import AsciiGridWriter
from envstretcher.batch_ascii_mapper import BatchAsciiMapper
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


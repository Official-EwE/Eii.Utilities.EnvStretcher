from __future__ import annotations

import shapefile

from diagnostics.triangulation_shapefile_exporter import (
    TriangulationShapefileExporter,
)
from envstretcher.batch_ascii_mapper import BatchAsciiMapper
from envstretcher.control_point import ControlPoint
from envstretcher.mapping_definition import MappingStrategy
from envstretcher.piecewise_affine_transform import PiecewiseAffineTransform
from envstretcher.raster_geometry import RasterGeometry


def _create_transform() -> PiecewiseAffineTransform:
    return PiecewiseAffineTransform(
        [
            ControlPoint(id="1", source_x=0.0, source_y=0.0, target_x=0.0, target_y=0.0),
            ControlPoint(id="2", source_x=10.0, source_y=0.0, target_x=6.0, target_y=0.0),
            ControlPoint(id="3", source_x=10.0, source_y=10.0, target_x=6.0, target_y=6.0),
            ControlPoint(id="4", source_x=0.0, source_y=10.0, target_x=0.0, target_y=6.0),
        ]
    )


def test_writes_source_and_target_shapefiles(tmp_path) -> None:
    transform = _create_transform()
    exporter = TriangulationShapefileExporter(transform)

    src_path = tmp_path / "source"
    tgt_path = tmp_path / "target"

    exporter.write(
        source_shapefile=src_path,
        target_shapefile=tgt_path,
    )

    assert (tmp_path / "source.shp").exists()
    assert (tmp_path / "source.dbf").exists()
    assert (tmp_path / "target.shp").exists()
    assert (tmp_path / "target.dbf").exists()


def test_feature_count_matches_triangles(tmp_path) -> None:
    transform = _create_transform()
    exporter = TriangulationShapefileExporter(transform)

    src_path = tmp_path / "source"
    tgt_path = tmp_path / "target"
    exporter.write(source_shapefile=src_path, target_shapefile=tgt_path)

    expected = len(transform.triangles)

    with shapefile.Reader(str(src_path)) as sf:
        assert len(sf.shapes()) == expected

    with shapefile.Reader(str(tgt_path)) as sf:
        assert len(sf.shapes()) == expected


def test_source_vertices_match_transform(tmp_path) -> None:
    transform = _create_transform()
    exporter = TriangulationShapefileExporter(transform)

    src_path = tmp_path / "source"
    exporter.write(source_shapefile=src_path, target_shapefile=tmp_path / "target")

    with shapefile.Reader(str(src_path)) as sf:
        for i, (shape, record) in enumerate(zip(sf.shapes(), sf.records())):
            triangle = transform.triangles[i]

            # pyshp returns [[x, y], ...] for POLYGON; last point closes ring
            ring = shape.points[:-1]
            for j, (x, y) in enumerate(ring):
                assert abs(x - triangle.source_vertices[j, 0]) < 1e-9
                assert abs(y - triangle.source_vertices[j, 1]) < 1e-9

            assert record["point_ids"] == ",".join(triangle.point_ids)
            assert record["id_a"] == triangle.point_ids[0]
            assert record["id_b"] == triangle.point_ids[1]
            assert record["id_c"] == triangle.point_ids[2]
            assert record["tri_number"] == i + 1


def test_target_vertices_match_transform(tmp_path) -> None:
    transform = _create_transform()
    exporter = TriangulationShapefileExporter(transform)

    tgt_path = tmp_path / "target"
    exporter.write(source_shapefile=tmp_path / "source", target_shapefile=tgt_path)

    with shapefile.Reader(str(tgt_path)) as sf:
        for i, shape in enumerate(sf.shapes()):
            triangle = transform.triangles[i]
            ring = shape.points[:-1]
            for j, (x, y) in enumerate(ring):
                assert abs(x - triangle.target_vertices[j, 0]) < 1e-9
                assert abs(y - triangle.target_vertices[j, 1]) < 1e-9


def test_creates_parent_directories(tmp_path) -> None:
    transform = _create_transform()
    exporter = TriangulationShapefileExporter(transform)

    exporter.write(
        source_shapefile=tmp_path / "shapefiles" / "source",
        target_shapefile=tmp_path / "shapefiles" / "target",
    )

    assert (tmp_path / "shapefiles" / "source.shp").exists()
    assert (tmp_path / "shapefiles" / "target.shp").exists()


def test_world_space_transform_produces_map_coordinates(tmp_path) -> None:
    # Index-space control points (col/row).
    index_transform = PiecewiseAffineTransform(
        [
            ControlPoint(id="1", source_x=0.0, source_y=0.0, target_x=0.0, target_y=0.0),
            ControlPoint(id="2", source_x=10.0, source_y=0.0, target_x=6.0, target_y=0.0),
            ControlPoint(id="3", source_x=10.0, source_y=10.0, target_x=6.0, target_y=6.0),
            ControlPoint(id="4", source_x=0.0, source_y=10.0, target_x=0.0, target_y=6.0),
        ]
    )

    source_geometry = RasterGeometry(
        rows=10, columns=10, x_min=-1.0, y_min=37.0, cell_size=0.1,
    )
    target_geometry = RasterGeometry(
        rows=6, columns=6, x_min=0.0, y_min=0.0, cell_size=0.2,
    )

    mapper = BatchAsciiMapper(
        transform=index_transform,
        target_geometry=target_geometry,
        strategy=MappingStrategy.CENTROID,
        use_index_control_points=True,
    )
    world_transform = mapper.build_world_space_transform(
        source_geometry=source_geometry,
        target_geometry=target_geometry,
    )

    exporter = TriangulationShapefileExporter(world_transform)
    src_path = tmp_path / "src"
    tgt_path = tmp_path / "tgt"
    exporter.write(source_shapefile=src_path, target_shapefile=tgt_path)

    # All source vertices should be within the source raster extent.
    with shapefile.Reader(str(src_path)) as sf:
        for shape in sf.shapes():
            for x, y in shape.points:
                assert source_geometry.x_min <= x <= source_geometry.x_max, \
                    f"source x={x} outside [{source_geometry.x_min}, {source_geometry.x_max}]"
                assert source_geometry.y_min <= y <= source_geometry.y_max, \
                    f"source y={y} outside [{source_geometry.y_min}, {source_geometry.y_max}]"

    # All target vertices should be within the target raster extent.
    with shapefile.Reader(str(tgt_path)) as sf:
        for shape in sf.shapes():
            for x, y in shape.points:
                assert target_geometry.x_min <= x <= target_geometry.x_max, \
                    f"target x={x} outside [{target_geometry.x_min}, {target_geometry.x_max}]"
                assert target_geometry.y_min <= y <= target_geometry.y_max, \
                    f"target y={y} outside [{target_geometry.y_min}, {target_geometry.y_max}]"

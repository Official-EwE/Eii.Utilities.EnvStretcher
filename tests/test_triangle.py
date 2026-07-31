import numpy as np
import pytest

from envstretcher.control_point import ControlPoint
from envstretcher.piecewise_affine_transform import PiecewiseAffineTransform
from envstretcher.triangle import Triangle


def test_triangle_contains_target_coordinate() -> None:
    triangle = Triangle(
        point_indices=(0, 1, 2),
        point_ids=("A", "B", "C"),
        source_vertices=np.asarray(
            [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            dtype=np.float64,
        ),
        target_vertices=np.asarray(
            [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            dtype=np.float64,
        ),
    )

    assert triangle.contains_target(0.5, 0.5)
    assert not triangle.contains_target(2.0, 2.0)


def test_triangle_maps_target_to_source() -> None:
    triangle = Triangle(
        point_indices=(0, 1, 2),
        point_ids=("A", "B", "C"),
        source_vertices=np.asarray(
            [[10.0, 10.0], [20.0, 10.0], [10.0, 20.0]],
            dtype=np.float64,
        ),
        target_vertices=np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float64,
        ),
    )

    source_x, source_y = triangle.target_to_source(0.25, 0.5)

    assert source_x == pytest.approx(12.5)
    assert source_y == pytest.approx(15.0)


def test_piecewise_transform_exposes_triangle_objects() -> None:
    control_points = [
        ControlPoint(id="1", source_x=0.0, source_y=0.0, target_x=0.0, target_y=0.0),
        ControlPoint(id="2", source_x=1.0, source_y=0.0, target_x=1.0, target_y=0.0),
        ControlPoint(id="3", source_x=0.0, source_y=1.0, target_x=0.0, target_y=1.0),
    ]

    transform = PiecewiseAffineTransform(
        control_points=control_points,
        triangles=(("1", "2", "3"),),
    )

    assert len(transform.triangles) == 1
    assert transform.triangles[0].point_ids == ("1", "2", "3")

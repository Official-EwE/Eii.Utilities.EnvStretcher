from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay

from .control_point import ControlPoint
from .mapping_result import MappingResult
from .triangle import Triangle


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class PiecewiseAffineTransform:
    """
    Piecewise affine mapping from target coordinates to source coordinates.

    Control points define corresponding locations in the target and source
    coordinate systems.

    If triangles are supplied, each triangle refers to three ControlPoint IDs.
    If triangles are omitted, a Delaunay triangulation is generated from the
    target control-point coordinates.
    """

    def __init__(
        self,
        control_points: Sequence[ControlPoint],
        triangles: Sequence[Sequence[str]] | None = None,
    ) -> None:
        self._control_points = tuple(control_points)

        self._validate_control_points()

        self._source_points = np.asarray(
            [point.source for point in self._control_points],
            dtype=np.float64,
        )

        self._target_points = np.asarray(
            [point.target for point in self._control_points],
            dtype=np.float64,
        )

        self._point_index_by_id = {
            point.id: index
            for index, point in enumerate(self._control_points)
        }

        if triangles is None:
            self._triangle_indices = self._create_delaunay_triangles()
        else:
            self._triangle_indices = self._resolve_triangle_ids(
                triangles
            )

        self._triangles = self._create_triangles(
            triangle_indices=self._triangle_indices
        )

    @property
    def control_points(self) -> tuple[ControlPoint, ...]:
        return self._control_points

    @property
    def source_points(self) -> FloatArray:
        return self._source_points.copy()

    @property
    def target_points(self) -> FloatArray:
        return self._target_points.copy()

    @property
    def triangle_indices(self) -> IntArray:
        return self._triangle_indices.copy()

    @property
    def triangle_ids(
        self,
    ) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            triangle.point_ids
            for triangle in self._triangles
        )

    @property
    def triangles(self) -> tuple[Triangle, ...]:
        return self._triangles

    def target_to_source(
        self,
        target_x: float,
        target_y: float,
    ) -> MappingResult:
        """
        Map one target coordinate to the corresponding source coordinate.

        Mapping fails when the coordinate lies outside all target triangles.
        """

        for triangle_index, triangle in enumerate(self._triangles):
            if not triangle.contains_target(
                target_x=target_x,
                target_y=target_y,
            ):
                continue

            source_x, source_y = triangle.target_to_source(
                target_x=target_x,
                target_y=target_y,
            )

            return MappingResult.succeeded(
                source_x=source_x,
                source_y=source_y,
                triangle_index=triangle_index,
            )

        return MappingResult.failed()

    def target_cell_to_source_by_max_overlap(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        sample_target_x: float | None = None,
        sample_target_y: float | None = None,
        tolerance: float = 1e-12,
    ) -> MappingResult:
        best_triangle_index = -1
        best_area = 0.0
        best_target_x = np.nan
        best_target_y = np.nan

        for triangle_index, triangle in enumerate(self._triangles):
            overlap = triangle.overlap_with_target_rectangle(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
            )

            if overlap is None:
                continue

            area, overlap_target_x, overlap_target_y = overlap

            if area <= best_area + tolerance:
                continue

            best_triangle_index = triangle_index
            best_area = area
            best_target_x = overlap_target_x
            best_target_y = overlap_target_y

        if best_triangle_index < 0:
            return MappingResult.failed()

        if sample_target_x is None or sample_target_y is None:
            sample_target_x = best_target_x
            sample_target_y = best_target_y

        source_x, source_y = self._triangles[
            best_triangle_index
        ].target_to_source(
            target_x=sample_target_x,
            target_y=sample_target_y,
        )

        return MappingResult.succeeded(
            source_x=source_x,
            source_y=source_y,
            triangle_index=best_triangle_index,
        )

    def _validate_control_points(self) -> None:
        if len(self._control_points) < 3:
            raise ValueError(
                "At least three control points are required."
            )

        id_counts = Counter(
            point.id
            for point in self._control_points
        )
        duplicate_ids = sorted(
            point_id
            for point_id, count in id_counts.items()
            if count > 1
        )

        if duplicate_ids:
            raise ValueError(
                "Control-point IDs must be unique. "
                f"Duplicate IDs: {duplicate_ids}"
            )

        coordinates = np.asarray(
            [
                [
                    point.source_x,
                    point.source_y,
                    point.target_x,
                    point.target_y,
                ]
                for point in self._control_points
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(coordinates)):
            raise ValueError(
                "Control-point coordinates must be finite."
            )

        source_locations = {
            point.source
            for point in self._control_points
        }

        if len(source_locations) != len(self._control_points):
            raise ValueError(
                "Duplicate source control-point coordinates found."
            )

        target_locations = {
            point.target
            for point in self._control_points
        }

        if len(target_locations) != len(self._control_points):
            raise ValueError(
                "Duplicate target control-point coordinates found."
            )

    def _create_delaunay_triangles(self) -> IntArray:
        try:
            triangulation = Delaunay(
                self._target_points
            )
        except Exception as error:
            raise ValueError(
                "Could not create a Delaunay triangulation "
                "from the target control points."
            ) from error

        return np.asarray(
            triangulation.simplices,
            dtype=np.int64,
        )

    def _resolve_triangle_ids(
        self,
        triangles: Sequence[Sequence[str]],
    ) -> IntArray:
        if len(triangles) == 0:
            raise ValueError(
                "The triangle collection cannot be empty."
            )

        resolved: list[tuple[int, int, int]] = []

        for triangle_number, triangle in enumerate(
            triangles,
            start=1,
        ):
            triangle_ids = tuple(str(value) for value in triangle)

            if len(triangle_ids) != 3:
                raise ValueError(
                    f"Triangle {triangle_number} must refer "
                    "to exactly three control-point IDs."
                )

            if len(set(triangle_ids)) != 3:
                raise ValueError(
                    f"Triangle {triangle_number} contains "
                    f"duplicate point IDs: {triangle_ids}"
                )

            unknown_ids = [
                point_id
                for point_id in triangle_ids
                if point_id not in self._point_index_by_id
            ]

            if unknown_ids:
                raise ValueError(
                    f"Triangle {triangle_number} refers to "
                    f"unknown control-point IDs: {unknown_ids}"
                )

            resolved.append(
                (
                    self._point_index_by_id[triangle_ids[0]],
                    self._point_index_by_id[triangle_ids[1]],
                    self._point_index_by_id[triangle_ids[2]],
                )
            )

        return np.asarray(
            resolved,
            dtype=np.int64,
        )

    def _create_triangles(
        self,
        triangle_indices: IntArray,
    ) -> tuple[Triangle, ...]:
        triangles = triangle_indices
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError(
                "Triangles must have shape (n, 3)."
            )

        if len(triangles) == 0:
            raise ValueError(
                "At least one triangle is required."
            )

        point_count = len(self._control_points)

        if np.any(triangles < 0):
            raise ValueError(
                "Triangle indices cannot be negative."
            )

        if np.any(triangles >= point_count):
            raise ValueError(
                "A triangle refers to a control-point index "
                "outside the control-point collection."
            )

        seen_triangles: set[tuple[int, int, int]] = set()
        result: list[Triangle] = []

        for triangle_number, triangle in enumerate(
            triangles,
            start=1,
        ):
            indices = tuple(int(value) for value in triangle)

            if len(set(indices)) != 3:
                raise ValueError(
                    f"Triangle {triangle_number} contains "
                    "duplicate control points."
                )

            canonical = tuple(sorted(indices))

            if canonical in seen_triangles:
                raise ValueError(
                    f"Triangle {triangle_number} duplicates "
                    "an earlier triangle."
                )

            seen_triangles.add(canonical)

            point_ids = tuple(
                self._control_points[index].id
                for index in indices
            )

            try:
                result.append(
                    Triangle(
                        point_indices=indices,
                        point_ids=point_ids,
                        source_vertices=self._source_points[
                            np.asarray(indices, dtype=np.int64)
                        ],
                        target_vertices=self._target_points[
                            np.asarray(indices, dtype=np.int64)
                        ],
                    )
                )
            except ValueError as error:
                raise ValueError(
                    f"Triangle {triangle_number} {point_ids} is invalid: "
                    f"{error}"
                ) from error

        return tuple(result)
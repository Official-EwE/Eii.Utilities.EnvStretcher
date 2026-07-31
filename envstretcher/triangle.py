from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .triangle_transform import TriangleTransform

FloatArray = NDArray[np.float64]


class Triangle:
    """
    One target/source triangle pair with cached affine transform.
    """

    def __init__(
        self,
        point_indices: tuple[int, int, int],
        point_ids: tuple[str, str, str],
        source_vertices: FloatArray,
        target_vertices: FloatArray,
        tolerance: float = 1e-12,
    ) -> None:
        self._point_indices = point_indices
        self._point_ids = point_ids
        self._source_vertices = np.asarray(
            source_vertices,
            dtype=np.float64,
        )
        self._target_vertices = np.asarray(
            target_vertices,
            dtype=np.float64,
        )
        self._tolerance = tolerance

        self._validate()
        self._transform = TriangleTransform(
            source_vertices=self._source_vertices,
            target_vertices=self._target_vertices,
        )

    @property
    def point_indices(self) -> tuple[int, int, int]:
        return self._point_indices

    @property
    def point_ids(self) -> tuple[str, str, str]:
        return self._point_ids

    @property
    def source_vertices(self) -> FloatArray:
        return self._source_vertices.copy()

    @property
    def target_vertices(self) -> FloatArray:
        return self._target_vertices.copy()

    def contains_target(
        self,
        target_x: float,
        target_y: float,
    ) -> bool:
        point = np.asarray(
            [target_x, target_y],
            dtype=np.float64,
        )

        a, b, c = self._target_vertices

        v0 = c - a
        v1 = b - a
        v2 = point - a

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        denominator = dot00 * dot11 - dot01 * dot01

        if abs(denominator) < self._tolerance:
            return False

        inv = 1.0 / denominator
        u = (dot11 * dot02 - dot01 * dot12) * inv
        v = (dot00 * dot12 - dot01 * dot02) * inv

        return (
            u >= -self._tolerance
            and v >= -self._tolerance
            and u + v <= 1.0 + self._tolerance
        )

    def target_to_source(
        self,
        target_x: float,
        target_y: float,
    ) -> tuple[float, float]:
        return self._transform.target_to_source(
            target_x=target_x,
            target_y=target_y,
        )

    def overlap_with_target_rectangle(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
    ) -> tuple[float, float, float] | None:
        polygon = [
            vertex.copy()
            for vertex in self._target_vertices
        ]

        polygon = self._clip_polygon_against_axis_bound(
            polygon=polygon,
            axis=0,
            bound=x_min,
            keep_greater=True,
        )
        polygon = self._clip_polygon_against_axis_bound(
            polygon=polygon,
            axis=0,
            bound=x_max,
            keep_greater=False,
        )
        polygon = self._clip_polygon_against_axis_bound(
            polygon=polygon,
            axis=1,
            bound=y_min,
            keep_greater=True,
        )
        polygon = self._clip_polygon_against_axis_bound(
            polygon=polygon,
            axis=1,
            bound=y_max,
            keep_greater=False,
        )

        if len(polygon) < 3:
            return None

        area, centroid_x, centroid_y = self._polygon_area_and_centroid(
            polygon
        )

        if area <= self._tolerance:
            return None

        return area, centroid_x, centroid_y

    def _clip_polygon_against_axis_bound(
        self,
        polygon: list[np.ndarray],
        axis: int,
        bound: float,
        keep_greater: bool,
    ) -> list[np.ndarray]:
        if not polygon:
            return []

        clipped: list[np.ndarray] = []
        previous = polygon[-1]

        for current in polygon:
            previous_inside = self._is_inside_half_plane(
                previous,
                axis=axis,
                bound=bound,
                keep_greater=keep_greater,
            )
            current_inside = self._is_inside_half_plane(
                current,
                axis=axis,
                bound=bound,
                keep_greater=keep_greater,
            )

            if current_inside:
                if not previous_inside:
                    clipped.append(
                        self._intersect_with_axis_bound(
                            previous,
                            current,
                            axis=axis,
                            bound=bound,
                        )
                    )

                clipped.append(current)

            elif previous_inside:
                clipped.append(
                    self._intersect_with_axis_bound(
                        previous,
                        current,
                        axis=axis,
                        bound=bound,
                    )
                )

            previous = current

        return clipped

    def _is_inside_half_plane(
        self,
        point: np.ndarray,
        axis: int,
        bound: float,
        keep_greater: bool,
    ) -> bool:
        coordinate = point[axis]

        if keep_greater:
            return bool(coordinate >= bound - self._tolerance)

        return bool(coordinate <= bound + self._tolerance)

    def _intersect_with_axis_bound(
        self,
        start: np.ndarray,
        end: np.ndarray,
        axis: int,
        bound: float,
    ) -> np.ndarray:
        delta = end[axis] - start[axis]

        if abs(delta) <= self._tolerance:
            result = end.copy()
            result[axis] = bound
            return result

        weight = float((bound - start[axis]) / delta)
        weight = float(np.clip(weight, 0.0, 1.0))

        result = start + weight * (end - start)
        result[axis] = bound
        return result

    def _polygon_area_and_centroid(
        self,
        polygon: list[np.ndarray],
    ) -> tuple[float, float, float]:
        vertices = np.asarray(
            polygon,
            dtype=np.float64,
        )

        x = vertices[:, 0]
        y = vertices[:, 1]

        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)

        cross = x * y_next - x_next * y
        double_area = float(np.sum(cross))

        if abs(double_area) <= self._tolerance:
            return 0.0, float("nan"), float("nan")

        area = 0.5 * abs(double_area)
        centroid_scale = 1.0 / (3.0 * double_area)

        centroid_x = float(np.sum((x + x_next) * cross) * centroid_scale)
        centroid_y = float(np.sum((y + y_next) * cross) * centroid_scale)

        return area, centroid_x, centroid_y

    def _validate(self) -> None:
        if len(self._point_indices) != 3:
            raise ValueError(
                "Triangle must contain exactly three point indices."
            )

        if len(set(self._point_indices)) != 3:
            raise ValueError(
                f"Triangle {self._point_ids} contains duplicate indices."
            )

        if any(index < 0 for index in self._point_indices):
            raise ValueError(
                f"Triangle {self._point_ids} contains negative indices."
            )

        if len(self._point_ids) != 3:
            raise ValueError(
                "Triangle must contain exactly three point IDs."
            )

        if len(set(self._point_ids)) != 3:
            raise ValueError(
                f"Triangle {self._point_ids} contains duplicate point IDs."
            )

        expected_shape = (3, 2)

        if self._source_vertices.shape != expected_shape:
            raise ValueError(
                "source_vertices must have shape (3, 2), "
                f"but received {self._source_vertices.shape}."
            )

        if self._target_vertices.shape != expected_shape:
            raise ValueError(
                "target_vertices must have shape (3, 2), "
                f"but received {self._target_vertices.shape}."
            )

        if not np.all(np.isfinite(self._source_vertices)):
            raise ValueError("source_vertices contains non-finite values.")

        if not np.all(np.isfinite(self._target_vertices)):
            raise ValueError("target_vertices contains non-finite values.")

        target_area = self._signed_area(self._target_vertices)
        source_area = self._signed_area(self._source_vertices)

        if abs(target_area) <= self._tolerance:
            raise ValueError(
                f"Target triangle {self._point_ids} is degenerate."
            )

        if abs(source_area) <= self._tolerance:
            raise ValueError(
                f"Source triangle {self._point_ids} is degenerate."
            )

        if np.sign(target_area) != np.sign(source_area):
            raise ValueError(
                f"Triangle {self._point_ids} is inverted between "
                "target and source."
            )

    @staticmethod
    def _signed_area(
        points: FloatArray,
    ) -> float:
        point_a, point_b, point_c = points

        return 0.5 * (
            (point_b[0] - point_a[0]) * (point_c[1] - point_a[1])
            - (point_b[1] - point_a[1]) * (point_c[0] - point_a[0])
        )

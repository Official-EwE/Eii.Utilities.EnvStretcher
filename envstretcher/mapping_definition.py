from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from envstretcher.control_point import ControlPoint
from envstretcher.piecewise_affine_transform import PiecewiseAffineTransform

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class MappingStrategy(StrEnum):
    CENTROID = "centroid"
    MAX_CELL_AREA = "max_cell_area"

    @classmethod
    def from_value(cls, value: str) -> MappingStrategy:
        try:
            return cls(value)
        except ValueError as error:
            valid_values = ", ".join(strategy.value for strategy in cls)
            raise ValueError(
                "Invalid mapping strategy "
                f"'{value}'. Expected one of: {valid_values}."
            ) from error


@dataclass(frozen=True)
class MappingDefinition:
    points: tuple[ControlPoint, ...]
    triangles: tuple[tuple[str, str, str], ...] | None = None
    mapping_strategy: MappingStrategy = MappingStrategy.CENTROID

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "points": [
                {
                    "id": point.id,
                    **(
                        {"name": point.name}
                        if point.name
                        else {}
                    ),
                    "source": [
                        point.source_x,
                        point.source_y,
                    ],
                    "target": [
                        point.target_x,
                        point.target_y,
                    ],
                }
                for point in self.points
            ],
            "mapping_strategy": self.mapping_strategy.value,
        }

        if self.triangles is not None:
            data["triangles"] = [
                list(triangle)
                for triangle in self.triangles
            ]

        return data

    @property
    def source_points(self) -> FloatArray:
        return np.asarray(
            [
                [point.source_x, point.source_y]
                for point in self.points
            ],
            dtype=np.float64,
        )

    @property
    def target_points(self) -> FloatArray:
        return np.asarray(
            [
                [point.target_x, point.target_y]
                for point in self.points
            ],
            dtype=np.float64,
        )

    def to_transform(self) -> PiecewiseAffineTransform:
        """Build a PiecewiseAffineTransform from this definition."""
        return PiecewiseAffineTransform(
            control_points=list(self.points),
            triangles=list(self.triangles) if self.triangles is not None else None,
        )

    @property
    def point_ids(self) -> tuple[str, ...]:
        return tuple(point.id for point in self.points)

    @property
    def triangle_indices(self) -> IntArray | None:
        if self.triangles is None:
            return None

        index_by_id = {
            point.id: index
            for index, point in enumerate(self.points)
        }

        return np.asarray(
            [
                [
                    index_by_id[point_id]
                    for point_id in triangle
                ]
                for triangle in self.triangles
            ],
            dtype=np.int64,
        )

    @classmethod
    def read(cls, filename: str | Path) -> MappingDefinition:
        filename = Path(filename)

        with filename.open("r", encoding="utf-8") as stream:
            data = json.load(stream)

        return cls.from_dict(data)

    def write(self, filename: str | Path) -> None:
        filename = Path(filename)

        with filename.open("w", encoding="utf-8") as stream:
            json.dump(
                self.to_dict(),
                stream,
                indent=2,
            )
            stream.write("\n")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MappingDefinition:
        raw_points = data.get("points")

        if not isinstance(raw_points, list):
            raise ValueError(
                "Mapping definition must contain a 'points' array."
            )

        if len(raw_points) < 3:
            raise ValueError(
                "At least three control points are required."
            )

        points: list[ControlPoint] = []

        for index, raw_point in enumerate(raw_points):
            if not isinstance(raw_point, dict):
                raise ValueError(
                    f"Point {index + 1} must be an object."
                )

            point_id = str(
                raw_point.get("id", index + 1)
            )
            point_name = str(
                raw_point.get("name", "")
            )

            source = cls._read_coordinate(
                raw_point,
                "source",
                point_id,
            )

            target = cls._read_coordinate(
                raw_point,
                "target",
                point_id,
            )

            points.append(
                ControlPoint(
                    id=point_id,
                    source_x=source[0],
                    source_y=source[1],
                    target_x=target[0],
                    target_y=target[1],
                    name=point_name,
                )
            )

        cls._validate_unique_ids(points)

        triangles = cls._read_triangles(
            data.get("triangles"),
            points,
        )

        mapping_strategy = cls._read_mapping_strategy(
            data.get("mapping_strategy")
        )

        return cls(
            points=tuple(points),
            triangles=triangles,
            mapping_strategy=mapping_strategy,
        )

    @staticmethod
    def _read_mapping_strategy(raw_strategy: Any) -> MappingStrategy:
        if raw_strategy is None:
            return MappingStrategy.CENTROID

        if not isinstance(raw_strategy, str):
            raise ValueError("'mapping_strategy' must be a string.")

        return MappingStrategy.from_value(raw_strategy)

    @staticmethod
    def _read_coordinate(
        raw_point: dict[str, Any],
        key: str,
        point_id: str,
    ) -> tuple[float, float]:
        coordinate = raw_point.get(key)

        if (
            not isinstance(coordinate, list)
            or len(coordinate) != 2
        ):
            raise ValueError(
                f"Point '{point_id}' must contain "
                f"'{key}': [x, y]."
            )

        try:
            return (
                float(coordinate[0]),
                float(coordinate[1]),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Point '{point_id}' contains an invalid "
                f"{key} coordinate."
            ) from error

    @staticmethod
    def _validate_unique_ids(
        points: list[ControlPoint],
    ) -> None:
        id_counts = Counter(point.id for point in points)
        duplicate_ids = sorted(
            point_id
            for point_id, count in id_counts.items()
            if count > 1
        )

        if duplicate_ids:
            raise ValueError(
                f"Duplicate control-point IDs: {duplicate_ids}"
            )

    @staticmethod
    def _read_triangles(
        raw_triangles: Any,
        points: list[ControlPoint],
    ) -> tuple[tuple[str, str, str], ...] | None:
        if raw_triangles is None:
            return None

        if not isinstance(raw_triangles, list):
            raise ValueError(
                "'triangles' must be an array."
            )

        known_ids = {
            point.id
            for point in points
        }

        triangles: list[tuple[str, str, str]] = []

        for index, raw_triangle in enumerate(raw_triangles):
            if (
                not isinstance(raw_triangle, list)
                or len(raw_triangle) != 3
            ):
                raise ValueError(
                    f"Triangle {index + 1} must contain "
                    f"exactly three point IDs."
                )

            triangle = tuple(
                str(point_id)
                for point_id in raw_triangle
            )

            if len(set(triangle)) != 3:
                raise ValueError(
                    f"Triangle {index + 1} contains "
                    f"duplicate point IDs: {triangle}"
                )

            unknown_ids = [
                point_id
                for point_id in triangle
                if point_id not in known_ids
            ]

            if unknown_ids:
                raise ValueError(
                    f"Triangle {index + 1} refers to "
                    f"unknown point IDs: {unknown_ids}"
                )

            triangles.append(triangle)

        if not triangles:
            raise ValueError(
                "'triangles' cannot be empty when provided."
            )

        return tuple(triangles)
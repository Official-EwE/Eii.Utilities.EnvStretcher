from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from envstretcher.bilinear_sampler import BilinearSampler
from envstretcher.mapping_definition import MappingStrategy
from envstretcher.piecewise_affine_transform import PiecewiseAffineTransform
from envstretcher.raster_geometry import RasterGeometry
from envstretcher.raster_grid import RasterGrid

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class MappingMatrix:
    """
    Precomputed mapping from target cells to source sample coordinates.

    The matrix captures expensive geometry work (triangle selection, overlap
    checks) so repeated warps can reuse it for many source rasters.
    """

    source_geometry: RasterGeometry
    target_geometry: RasterGeometry
    strategy: MappingStrategy

    source_x: FloatArray
    source_y: FloatArray
    success: BoolArray
    triangle_index: IntArray

    def __post_init__(self) -> None:
        expected_shape = (
            self.target_geometry.rows,
            self.target_geometry.columns,
        )

        for name, values in (
            ("source_x", self.source_x),
            ("source_y", self.source_y),
            ("success", self.success),
            ("triangle_index", self.triangle_index),
        ):
            if values.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, "
                    f"but received {values.shape}."
                )

    @classmethod
    def build(
        cls,
        transform: PiecewiseAffineTransform,
        source_geometry: RasterGeometry,
        target_geometry: RasterGeometry,
        strategy: MappingStrategy = MappingStrategy.CENTROID,
    ) -> MappingMatrix:
        source_x = np.full(
            (target_geometry.rows, target_geometry.columns),
            np.nan,
            dtype=np.float64,
        )
        source_y = np.full_like(source_x, np.nan)
        success = np.zeros_like(source_x, dtype=bool)
        triangle_index = np.full_like(source_x, -1, dtype=np.int64)

        for row in range(target_geometry.rows):
            for column in range(target_geometry.columns):
                if strategy == MappingStrategy.CENTROID:
                    target_x, target_y = target_geometry.cell_center(
                        row=row,
                        column=column,
                    )

                    mapping = transform.target_to_source(
                        target_x=target_x,
                        target_y=target_y,
                    )

                elif strategy == MappingStrategy.MAX_CELL_AREA:
                    target_x, target_y = target_geometry.cell_center(
                        row=row,
                        column=column,
                    )

                    cell_size = target_geometry.cell_size
                    x_min = target_geometry.x_min + column * cell_size
                    x_max = x_min + cell_size

                    y_max = target_geometry.y_max - row * cell_size
                    y_min = y_max - cell_size

                    mapping = transform.target_cell_to_source_by_max_overlap(
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max,
                        sample_target_x=target_x,
                        sample_target_y=target_y,
                    )

                else:
                    raise ValueError(
                        "Unsupported mapping strategy: "
                        f"{strategy}"
                    )

                if not mapping.success:
                    continue

                success[row, column] = True
                source_x[row, column] = mapping.source_x
                source_y[row, column] = mapping.source_y
                triangle_index[row, column] = mapping.triangle_index

        return cls(
            source_geometry=source_geometry,
            target_geometry=target_geometry,
            strategy=strategy,
            source_x=source_x,
            source_y=source_y,
            success=success,
            triangle_index=triangle_index,
        )

    def is_compatible_source_geometry(
        self,
        geometry: RasterGeometry,
    ) -> bool:
        return geometry == self.source_geometry

    def warp(
        self,
        sampler: BilinearSampler,
        nodata: float = np.nan,
    ) -> RasterGrid:
        if not self.is_compatible_source_geometry(sampler.grid.geometry):
            raise ValueError(
                "Source geometry mismatch: mapping matrix was built for "
                f"{self.source_geometry}, got {sampler.grid.geometry}."
            )

        target = RasterGrid.create_empty(
            geometry=self.target_geometry,
            nodata=nodata,
        )

        for row in range(self.target_geometry.rows):
            for column in range(self.target_geometry.columns):
                if not self.success[row, column]:
                    continue

                target.values[row, column] = sampler.sample(
                    x=float(self.source_x[row, column]),
                    y=float(self.source_y[row, column]),
                )

        return target

    def write(self, filename: str | Path) -> None:
        filename = Path(filename)

        np.savez_compressed(
            filename,
            source_x=self.source_x,
            source_y=self.source_y,
            success=self.success,
            triangle_index=self.triangle_index,
            strategy=np.asarray(self.strategy.value),
            source_rows=np.asarray(self.source_geometry.rows),
            source_columns=np.asarray(self.source_geometry.columns),
            source_x_min=np.asarray(self.source_geometry.x_min),
            source_y_min=np.asarray(self.source_geometry.y_min),
            source_cell_size=np.asarray(self.source_geometry.cell_size),
            target_rows=np.asarray(self.target_geometry.rows),
            target_columns=np.asarray(self.target_geometry.columns),
            target_x_min=np.asarray(self.target_geometry.x_min),
            target_y_min=np.asarray(self.target_geometry.y_min),
            target_cell_size=np.asarray(self.target_geometry.cell_size),
        )

    @classmethod
    def read(cls, filename: str | Path) -> MappingMatrix:
        filename = Path(filename)

        with np.load(filename, allow_pickle=False) as data:
            source_geometry = RasterGeometry(
                rows=int(data["source_rows"]),
                columns=int(data["source_columns"]),
                x_min=float(data["source_x_min"]),
                y_min=float(data["source_y_min"]),
                cell_size=float(data["source_cell_size"]),
            )

            target_geometry = RasterGeometry(
                rows=int(data["target_rows"]),
                columns=int(data["target_columns"]),
                x_min=float(data["target_x_min"]),
                y_min=float(data["target_y_min"]),
                cell_size=float(data["target_cell_size"]),
            )

            strategy = MappingStrategy.from_value(str(data["strategy"]))

            return cls(
                source_geometry=source_geometry,
                target_geometry=target_geometry,
                strategy=strategy,
                source_x=np.asarray(data["source_x"], dtype=np.float64),
                source_y=np.asarray(data["source_y"], dtype=np.float64),
                success=np.asarray(data["success"], dtype=bool),
                triangle_index=np.asarray(
                    data["triangle_index"],
                    dtype=np.int64,
                ),
            )


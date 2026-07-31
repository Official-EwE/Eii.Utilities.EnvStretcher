from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from envstretcher.raster_geometry import RasterGeometry

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class RasterGrid:
    """
    Raster values associated with a RasterGeometry.

    Geometry is immutable, while values remain mutable so that algorithms
    such as raster warping can fill an output grid in place.
    """

    geometry: RasterGeometry
    values: FloatArray
    nodata: float = np.nan

    def __post_init__(self) -> None:
        self.values = np.asarray(
            self.values,
            dtype=np.float64,
        )

        expected_shape = (
            self.geometry.rows,
            self.geometry.columns,
        )

        if self.values.shape != expected_shape:
            raise ValueError(
                f"Raster values must have shape {expected_shape}, "
                f"but received {self.values.shape}."
            )

    @classmethod
    def create_empty(
        cls,
        geometry: RasterGeometry,
        nodata: float = np.nan,
    ) -> RasterGrid:
        """Create a raster whose cells are initially filled with nodata."""

        values = np.full(
            shape=(geometry.rows, geometry.columns),
            fill_value=nodata,
            dtype=np.float64,
        )

        return cls(
            geometry=geometry,
            values=values,
            nodata=nodata,
        )

    @classmethod
    def create_filled(
        cls,
        geometry: RasterGeometry,
        value: float,
        nodata: float = np.nan,
    ) -> RasterGrid:
        """Create a raster whose cells are filled with one value."""

        if not np.isfinite(value):
            raise ValueError("The fill value must be finite.")

        values = np.full(
            shape=(geometry.rows, geometry.columns),
            fill_value=value,
            dtype=np.float64,
        )

        return cls(
            geometry=geometry,
            values=values,
            nodata=nodata,
        )

    @property
    def rows(self) -> int:
        return self.geometry.rows

    @property
    def columns(self) -> int:
        return self.geometry.columns

    def cell_center(
        self,
        row: int,
        column: int,
    ) -> tuple[float, float]:
        """Return the centre coordinate of one raster cell."""

        return self.geometry.cell_center(row, column)

    def is_nodata(self, value: float) -> bool:
        """Return True when a value represents nodata."""

        if np.isnan(self.nodata):
            return bool(np.isnan(value))

        return bool(
            np.isnan(value)
            or np.isclose(value, self.nodata)
        )
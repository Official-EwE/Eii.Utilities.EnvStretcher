from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RasterGeometry:
    """
    Spatial geometry of a regular square-cell raster.

    Conventions
    -----------
    - x_min and y_min describe the outer lower-left corner.
    - Row 0 is the uppermost/northern row.
    - Column 0 is the leftmost/western column.
    - Coordinates returned by cell_center() refer to cell centres.
    """

    rows: int
    columns: int

    x_min: float
    y_min: float

    cell_size: float

    def __post_init__(self) -> None:
        if self.rows <= 0:
            raise ValueError("rows must be greater than zero.")

        if self.columns <= 0:
            raise ValueError("columns must be greater than zero.")

        if not np.isfinite(self.x_min):
            raise ValueError("x_min must be finite.")

        if not np.isfinite(self.y_min):
            raise ValueError("y_min must be finite.")

        if not np.isfinite(self.cell_size):
            raise ValueError("cell_size must be finite.")

        if self.cell_size <= 0.0:
            raise ValueError("cell_size must be greater than zero.")

    @property
    def x_max(self) -> float:
        """Outer right-hand boundary of the raster."""

        return self.x_min + self.columns * self.cell_size

    @property
    def y_max(self) -> float:
        """Outer upper boundary of the raster."""

        return self.y_min + self.rows * self.cell_size

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """
        Return the outer raster extent as:

            x_min, y_min, x_max, y_max
        """

        return (
            self.x_min,
            self.y_min,
            self.x_max,
            self.y_max,
        )

    def contains_coordinate(
        self,
        x: float,
        y: float,
    ) -> bool:
        """
        Return True when a coordinate lies inside the raster footprint.

        The outer maximum boundaries are included to tolerate coordinates
        produced by floating-point calculations.
        """

        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
        )

    def contains_cell(
        self,
        row: int,
        column: int,
    ) -> bool:
        """Return True when a row and column identify an existing cell."""

        return (
            0 <= row < self.rows
            and 0 <= column < self.columns
        )

    def cell_center(
        self,
        row: int,
        column: int,
    ) -> tuple[float, float]:
        """
        Return the centre coordinate of one raster cell.

        Row 0 is the top row, so y decreases as row increases.
        """

        if not self.contains_cell(row, column):
            raise IndexError(
                f"Cell ({row}, {column}) is outside a raster with "
                f"shape ({self.rows}, {self.columns})."
            )

        x = self.x_min + (column + 0.5) * self.cell_size
        y = self.y_max - (row + 0.5) * self.cell_size

        return x, y

    def fractional_cell(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        """
        Convert a coordinate to fractional row and column indices.

        Integer row and column values correspond to cell centres.

        For example:

            row = 0.0, column = 0.0

        identifies the centre of the upper-left cell.
        """

        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError("Coordinates must be finite.")

        column = (x - self.x_min) / self.cell_size - 0.5
        row = (self.y_max - y) / self.cell_size - 0.5

        return row, column
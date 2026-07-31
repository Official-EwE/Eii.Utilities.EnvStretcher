import numpy as np

from envstretcher.raster_grid import RasterGrid


class BilinearSampler:
    """
    Sample continuous coordinates from a RasterGrid using bilinear
    interpolation.

    Coordinates inside the outer half-cell margins are clamped to the
    nearest edge-cell centre. Coordinates outside the raster footprint
    return nodata.

    If a contributing source cell contains nodata, the sample also returns
    nodata.
    """

    def __init__(
        self,
        grid: RasterGrid,
    ) -> None:
        self._grid = grid

    @property
    def grid(self) -> RasterGrid:
        return self._grid

    def is_nodata(self, value) -> bool:
        return self._grid.is_nodata(value)

    def sample(
        self,
        x: float,
        y: float,
    ) -> float:
        """
        Sample the raster at one coordinate.

        Returns the grid's nodata value when:
        - the coordinate lies outside the raster footprint; or
        - a cell contributing to the interpolation contains nodata.
        """

        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError("Sample coordinates must be finite.")

        geometry = self._grid.geometry

        if not geometry.contains_coordinate(x, y):
            return float(self._grid.nodata)

        fractional_row, fractional_column = (
            geometry.fractional_cell(x, y)
        )

        # Coordinates in the half-cell margin around the raster are
        # represented by the nearest edge-cell value.
        fractional_row = float(
            np.clip(
                fractional_row,
                0.0,
                geometry.rows - 1.0,
            )
        )

        fractional_column = float(
            np.clip(
                fractional_column,
                0.0,
                geometry.columns - 1.0,
            )
        )

        row0 = int(np.floor(fractional_row))
        column0 = int(np.floor(fractional_column))

        row1 = min(row0 + 1, geometry.rows - 1)
        column1 = min(column0 + 1, geometry.columns - 1)

        row_fraction = fractional_row - row0
        column_fraction = fractional_column - column0

        values = self._grid.values

        samples = (
            (
                values[row0, column0],
                (1.0 - row_fraction)
                * (1.0 - column_fraction),
            ),
            (
                values[row0, column1],
                (1.0 - row_fraction)
                * column_fraction,
            ),
            (
                values[row1, column0],
                row_fraction
                * (1.0 - column_fraction),
            ),
            (
                values[row1, column1],
                row_fraction
                * column_fraction,
            ),
        )

        result = 0.0

        for value, weight in samples:
            # A zero-weight neighbour does not contribute and therefore
            # should not invalidate an otherwise valid sample.
            if weight <= 0.0:
                continue

            if self._grid.is_nodata(float(value)):
                return float(self._grid.nodata)

            result += weight * float(value)

        return result
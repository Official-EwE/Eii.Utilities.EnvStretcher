import numpy as np

from envstretcher.raster_grid import RasterGrid


class BilinearSampler:
    """
    Sample continuous coordinates from a RasterGrid using bilinear
    interpolation.

    Coordinates inside the outer half-cell margins are clamped to the
    nearest edge-cell centre. Coordinates outside the raster footprint
    return nodata.

    During interpolation, source nodata values (and optionally non-positive
    source values used as a land mask) are excluded and weights are
    renormalized over the remaining valid neighbours.
    """

    def __init__(
        self,
        grid: RasterGrid,
        treat_nonpositive_as_nodata: bool = True,
    ) -> None:
        self._grid = grid
        self._treat_nonpositive_as_nodata = treat_nonpositive_as_nodata

    @property
    def grid(self) -> RasterGrid:
        return self._grid

    def is_nodata(self, value) -> bool:
        return self._grid.is_nodata(value)

    def _is_invalid_source_value(self, value: float) -> bool:
        if self._grid.is_nodata(value):
            return True

        return bool(
            self._treat_nonpositive_as_nodata
            and value <= 0.0
        )

    def sample(
        self,
        x: float,
        y: float,
    ) -> float:
        """
        Sample the raster at one coordinate.

        Returns the grid's nodata value when:
        - the coordinate lies outside the raster footprint; or
        - all contributing neighbours are invalid.
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

        weighted_sum = 0.0
        valid_weight_sum = 0.0

        for value, weight in samples:
            # A zero-weight neighbour does not contribute and therefore
            # should not invalidate an otherwise valid sample.
            if weight <= 0.0:
                continue

            source_value = float(value)

            if self._is_invalid_source_value(source_value):
                continue

            weighted_sum += weight * source_value
            valid_weight_sum += weight

        if valid_weight_sum <= 0.0:
            return float(self._grid.nodata)

        return weighted_sum / valid_weight_sum
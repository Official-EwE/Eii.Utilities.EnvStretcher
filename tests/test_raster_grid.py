import numpy as np
import pytest

from envstretcher.raster_geometry import RasterGeometry
from envstretcher.raster_grid import RasterGrid


def test_raster_grid_rejects_wrong_shape() -> None:
    geometry = RasterGeometry(
        rows=2,
        columns=3,
        x_min=0.0,
        y_min=0.0,
        cell_size=1.0,
    )

    values = np.zeros((3, 2))

    with pytest.raises(ValueError):
        RasterGrid(
            geometry=geometry,
            values=values,
            nodata=-9999.0,
        )
from __future__ import annotations

import numpy as np

from envstretcher.bilinear_sampler import BilinearSampler
from envstretcher.raster_geometry import RasterGeometry
from envstretcher.raster_grid import RasterGrid


def test_bilinear_sampler_excludes_invalid_neighbours_and_renormalizes() -> None:
    geometry = RasterGeometry(
        rows=2,
        columns=2,
        x_min=0.0,
        y_min=0.0,
        cell_size=1.0,
    )
    grid = RasterGrid(
        geometry=geometry,
        values=np.asarray(
            [
                [10.0, 0.0],
                [30.0, 50.0],
            ],
            dtype=np.float64,
        ),
        nodata=-9999.0,
    )
    sampler = BilinearSampler(grid)

    sampled = sampler.sample(1.0, 1.0)

    assert sampled == 30.0


def test_bilinear_sampler_returns_nodata_when_all_neighbours_invalid() -> None:
    geometry = RasterGeometry(
        rows=2,
        columns=2,
        x_min=0.0,
        y_min=0.0,
        cell_size=1.0,
    )
    grid = RasterGrid(
        geometry=geometry,
        values=np.asarray(
            [
                [0.0, -9999.0],
                [np.nan, 0.0],
            ],
            dtype=np.float64,
        ),
        nodata=-9999.0,
    )
    sampler = BilinearSampler(grid)

    sampled = sampler.sample(1.0, 1.0)

    assert grid.is_nodata(sampled)

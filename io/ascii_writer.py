from __future__ import annotations

from pathlib import Path

import numpy as np

from envstretcher.raster_grid import RasterGrid


class AsciiGridWriter:
    """Write a RasterGrid to an ESRI ASCII raster."""

    @staticmethod
    def write(
        raster: RasterGrid,
        filename: str | Path,
        decimals: int = 6,
    ) -> None:
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        geometry = raster.geometry

        with filename.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(f"ncols         {geometry.columns}\n")
            stream.write(f"nrows         {geometry.rows}\n")
            stream.write(f"xllcorner     {geometry.x_min:.12g}\n")
            stream.write(f"yllcorner     {geometry.y_min:.12g}\n")
            stream.write(f"cellsize      {geometry.cell_size:.12g}\n")
            stream.write(f"NODATA_value  {raster.nodata:.12g}\n")

            fmt = f"%.{decimals}f"

            for row in raster.values:
                np.savetxt(
                    stream,
                    row.reshape(1, -1),
                    fmt=fmt,
                    delimiter=" ",
                )


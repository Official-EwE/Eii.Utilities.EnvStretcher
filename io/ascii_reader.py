from __future__ import annotations

from pathlib import Path

import numpy as np

from envstretcher.raster_geometry import RasterGeometry
from envstretcher.raster_grid import RasterGrid


class AsciiGridReader:
    """Read an ESRI ASCII raster into a RasterGrid."""

    REQUIRED = {
        "ncols",
        "nrows",
        "cellsize",
    }

    @staticmethod
    def read(
        filename: str | Path,
    ) -> RasterGrid:
        filename = Path(filename)

        header: dict[str, str] = {}

        with filename.open("r", encoding="utf-8") as stream:
            for _ in range(6):
                line = stream.readline()

                if not line:
                    raise ValueError("Unexpected end of header.")

                parts = line.strip().split()
                if len(parts) < 2:
                    raise ValueError(f"Invalid header line: {line}")

                header[parts[0].lower()] = parts[1]

            missing = AsciiGridReader.REQUIRED - set(header)
            if missing:
                raise ValueError(
                    f"Missing required header entries: {sorted(missing)}"
                )

            ncols = int(header["ncols"])
            nrows = int(header["nrows"])
            cell_size = float(header["cellsize"])
            nodata = float(header.get("nodata_value", -9999.0))

            if "xllcorner" in header and "yllcorner" in header:
                x_min = float(header["xllcorner"])
                y_min = float(header["yllcorner"])
            elif "xllcenter" in header and "yllcenter" in header:
                x_min = float(header["xllcenter"]) - cell_size / 2.0
                y_min = float(header["yllcenter"]) - cell_size / 2.0
            else:
                raise ValueError(
                    "Header must contain either xllcorner/yllcorner "
                    "or xllcenter/yllcenter."
                )

            values = np.loadtxt(stream, dtype=np.float64)

        if values.shape != (nrows, ncols):
            raise ValueError(
                f"Expected raster of shape ({nrows}, {ncols}), "
                f"found {values.shape}."
            )

        geometry = RasterGeometry(
            rows=nrows,
            columns=ncols,
            x_min=x_min,
            y_min=y_min,
            cell_size=cell_size,
        )

        return RasterGrid(
            geometry=geometry,
            values=values,
            nodata=nodata,
        )


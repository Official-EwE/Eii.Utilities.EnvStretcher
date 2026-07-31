import numpy as np

from diagnostics.coverage_summary import CoverageSummary
from envstretcher.bilinear_sampler import BilinearSampler
from envstretcher.piecewise_affine_transform import PiecewiseAffineTransform
from envstretcher.raster_grid import RasterGrid

from envstretcher.coverage_code import CoverageCode



class MappingDiagnostics:
    """
    Produces a diagnostic raster describing how each target cell
    was classified during mapping.

    The output raster contains CoverageCode values.
    """

    def __init__(
        self,
        transform: PiecewiseAffineTransform,
        sampler: BilinearSampler,
    ):

        self._transform = transform
        self._sampler = sampler

    def classify(
        self,
        basemap: RasterGrid,
    ) -> RasterGrid:
        output = RasterGrid.create_empty(
            geometry=basemap.geometry,
            nodata=0,
        )

        for row in range(basemap.rows):

            for col in range(basemap.columns):

                depth = basemap.values[row, col]

                # -------------------------------------------------
                # Land
                # -------------------------------------------------

                if basemap.is_nodata(depth) or depth <= 0:

                    output.values[row, col] = CoverageCode.LAND
                    continue

                # -------------------------------------------------
                # Map target -> source
                # -------------------------------------------------

                x, y = basemap.geometry.cell_center(row, col)

                mapping = self._transform.target_to_source(x, y)

                if not mapping.success:

                    output.values[row, col] = CoverageCode.OUTSIDE_MAPPING
                    continue

                # -------------------------------------------------
                # Sample source raster
                # -------------------------------------------------

                value = self._sampler.sample(
                    mapping.source_x,
                    mapping.source_y,
                )

                if self._sampler.is_nodata(value):

                    output.values[row, col] = CoverageCode.SOURCE_NODATA

                else:

                    output.values[row, col] = CoverageCode.VALID_DATA

        return output

    def classify_warped(
        self,
        warped: RasterGrid,
        basemap: RasterGrid,
    ) -> RasterGrid:
        if warped.geometry != basemap.geometry:
            raise ValueError(
                "Warped and basemap geometries must match."
            )

        output = RasterGrid.create_empty(
            geometry=basemap.geometry,
            nodata=0,
        )

        for row in range(basemap.rows):
            for col in range(basemap.columns):
                depth = basemap.values[row, col]

                if basemap.is_nodata(depth) or depth <= 0:
                    output.values[row, col] = CoverageCode.LAND
                    continue

                warped_value = warped.values[row, col]

                if self._sampler.is_nodata(warped_value):
                    output.values[row, col] = CoverageCode.OUTSIDE_MAPPING
                else:
                    output.values[row, col] = CoverageCode.VALID_DATA

        return output

    def summarize(
            self,
            diagnostics: RasterGrid,
    ) -> CoverageSummary:

        values = diagnostics.values

        land = np.count_nonzero(values == CoverageCode.LAND)
        mapped = np.count_nonzero(values == CoverageCode.VALID_DATA)
        nodata = np.count_nonzero(values == CoverageCode.SOURCE_NODATA)
        outside = np.count_nonzero(values == CoverageCode.OUTSIDE_MAPPING)

        water = mapped + nodata + outside

        return CoverageSummary(
            land_cells=land,
            water_cells=water,
            mapped_cells=mapped,
            source_nodata_cells=nodata,
            outside_mapping_cells=outside,
        )
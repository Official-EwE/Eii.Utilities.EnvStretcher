from enum import IntEnum


class CoverageCode(IntEnum):
    """
    Classification assigned to each target cell during mapping diagnostics.

    LAND
        Target basemap cell is land (depth <= 0). No mapping attempted.

    VALID_DATA
        Target cell mapped successfully and a valid source value was obtained.

    SOURCE_NODATA
        Target cell mapped successfully, but the source raster could not
        provide a valid value.

    OUTSIDE_MAPPING
        Target cell centre lies outside the target control-point triangulation.
    """

    LAND = 1

    VALID_DATA = 2

    SOURCE_NODATA = 3

    OUTSIDE_MAPPING = 4

    @property
    def description(self) -> str:

        return {
            CoverageCode.LAND: "Land",
            CoverageCode.VALID_DATA: "Mapped successfully",
            CoverageCode.SOURCE_NODATA: "Mapped, but source has no data",
            CoverageCode.OUTSIDE_MAPPING: "Outside target mapping",
        }[self]
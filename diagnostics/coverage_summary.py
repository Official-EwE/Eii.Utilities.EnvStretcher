from dataclasses import dataclass


@dataclass(frozen=True)
class CoverageSummary:
    land_cells: int
    water_cells: int
    mapped_cells: int
    source_nodata_cells: int
    outside_mapping_cells: int

    @property
    def all_water_cells_covered(self) -> bool:
        return (
            self.source_nodata_cells == 0
            and self.outside_mapping_cells == 0
        )

    @property
    def coverage_fraction(self) -> float:
        if self.water_cells == 0:
            return 1.0
        return self.mapped_cells / self.water_cells

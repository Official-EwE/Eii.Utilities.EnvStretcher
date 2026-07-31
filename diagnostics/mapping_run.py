from dataclasses import dataclass


@dataclass(slots=True)
class MappingRun:
    envstretcher_version: str
    transform: str
    sampler: str
    source_raster: str
    target_basemap: str
    source_points: str
    target_points: str
    output_raster: str
    diagnostics_raster: str
from __future__ import annotations
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from io.ascii_reader import AsciiGridReader
from io.ascii_writer import AsciiGridWriter
from envstretcher.bilinear_sampler import BilinearSampler
from envstretcher.coverage_code import CoverageCode
from envstretcher.mapping_definition import MappingStrategy
from envstretcher.mapping_matrix import MappingMatrix
from envstretcher.piecewise_affine_transform import PiecewiseAffineTransform
from envstretcher.raster_geometry import RasterGeometry
from envstretcher.raster_grid import RasterGrid
from envstretcher.raster_warper import RasterWarper


@dataclass(frozen=True, slots=True)
class BatchWarpStats:
    processed_files: int


class BatchAsciiMapper:
    """
    Two-phase helper for ESRI ASCII batch workflows.

    Phase 1: build mapping matrix once from a template source raster.
    Phase 2: reuse mapping matrix for any source raster with matching geometry.
    """

    def __init__(
        self,
        transform: PiecewiseAffineTransform,
        target_geometry: RasterGeometry,
        strategy: MappingStrategy = MappingStrategy.CENTROID,
    ) -> None:
        self._transform = transform
        self._target_geometry = target_geometry
        self._strategy = strategy

    def build_mapping_matrix(
        self,
        template_source_ascii: str | Path,
        output_matrix_file: str | Path | None = None,
    ) -> MappingMatrix:
        template_grid = AsciiGridReader.read(template_source_ascii)

        warper = RasterWarper(
            sampler=BilinearSampler(template_grid),
            transform=self._transform,
        )

        matrix = warper.build_mapping_matrix(
            target_geometry=self._target_geometry,
            strategy=self._strategy,
        )

        if output_matrix_file is not None:
            matrix.write(output_matrix_file)

        return matrix

    def warp_files(
        self,
        source_files: Iterable[str | Path],
        output_directory: str | Path,
        matrix: MappingMatrix,
        decimals: int = 6,
    ) -> BatchWarpStats:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        processed = 0

        for source_file in source_files:
            source_file = Path(source_file)
            source_grid = AsciiGridReader.read(source_file)

            warped = matrix.warp(
                sampler=BilinearSampler(source_grid),
            )

            output_path = output_directory / source_file.name
            AsciiGridWriter.write(
                raster=warped,
                filename=output_path,
                decimals=decimals,
            )

            processed += 1

        return BatchWarpStats(processed_files=processed)

    def warp_glob(
        self,
        source_directory: str | Path,
        pattern: str,
        output_directory: str | Path,
        matrix: MappingMatrix,
        decimals: int = 6,
    ) -> BatchWarpStats:
        source_directory = Path(source_directory)
        files = sorted(source_directory.glob(pattern))

        return self.warp_files(
            source_files=files,
            output_directory=output_directory,
            matrix=matrix,
            decimals=decimals,
        )

    def warp_glob_with_diagnostics(
        self,
        source_directory: str | Path,
        pattern: str,
        output_directory: str | Path,
        matrix: MappingMatrix,
        basemap: RasterGrid,
        diagnostics_directory: str | Path | None = None,
        decimals: int = 6,
    ) -> BatchWarpStats:
        source_directory = Path(source_directory)
        files = sorted(source_directory.glob(pattern))

        return self.warp_files_with_diagnostics(
            source_files=files,
            output_directory=output_directory,
            matrix=matrix,
            basemap=basemap,
            diagnostics_directory=diagnostics_directory,
            decimals=decimals,
        )

    def warp_files_with_diagnostics(
        self,
        source_files: Iterable[str | Path],
        output_directory: str | Path,
        matrix: MappingMatrix,
        basemap: RasterGrid,
        diagnostics_directory: str | Path | None = None,
        decimals: int = 6,
    ) -> BatchWarpStats:
        import numpy as np

        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        diagnostics_directory = (
            Path(diagnostics_directory)
            if diagnostics_directory is not None
            else None
        )
        if diagnostics_directory is not None:
            diagnostics_directory.mkdir(parents=True, exist_ok=True)

        processed = 0

        for source_file in source_files:
            source_file = Path(source_file)
            source_grid = AsciiGridReader.read(source_file)

            warped = matrix.warp(
                sampler=BilinearSampler(source_grid),
            )

            output_path = output_directory / source_file.name
            AsciiGridWriter.write(
                raster=warped,
                filename=output_path,
                decimals=decimals,
            )

            if diagnostics_directory is not None:
                diag_grid = RasterGrid.create_empty(
                    geometry=basemap.geometry,
                    nodata=0,
                )

                for row in range(basemap.rows):
                    for col in range(basemap.columns):
                        depth = basemap.values[row, col]

                        if basemap.is_nodata(depth) or depth <= 0:
                            diag_grid.values[row, col] = CoverageCode.LAND
                            continue

                        warped_value = warped.values[row, col]

                        if np.isnan(warped_value):
                            diag_grid.values[row, col] = (
                                CoverageCode.OUTSIDE_MAPPING
                            )
                        else:
                            diag_grid.values[row, col] = (
                                CoverageCode.VALID_DATA
                            )

                diag_path = diagnostics_directory / (
                    source_file.stem + "_diag.asc"
                )
                AsciiGridWriter.write(
                    raster=diag_grid,
                    filename=diag_path,
                )

            processed += 1

        return BatchWarpStats(processed_files=processed)


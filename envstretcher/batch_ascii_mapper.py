from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from diagnostics.coverage_summary import CoverageSummary
from diagnostics.mapping_diagnostics import MappingDiagnostics
from envstretcher.control_point import ControlPoint
from envstretcher.io.ascii_reader import AsciiGridReader
from envstretcher.io.ascii_writer import AsciiGridWriter
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
        use_index_control_points: bool = False,
    ) -> None:
        self._transform = transform
        self._target_geometry = target_geometry
        self._strategy = strategy
        self._use_index_control_points = use_index_control_points

    def build_mapping_matrix(
        self,
        template_source_ascii: str | Path,
        output_matrix_file: str | Path | None = None,
    ) -> MappingMatrix:
        template_grid = AsciiGridReader.read(template_source_ascii)
        transform = self._create_transform_for_geometries(
            source_geometry=template_grid.geometry,
            target_geometry=self._target_geometry,
        )

        warper = RasterWarper(
            sampler=BilinearSampler(template_grid),
            transform=transform,
        )

        matrix = warper.build_mapping_matrix(
            target_geometry=self._target_geometry,
            strategy=self._strategy,
        )

        if output_matrix_file is not None:
            matrix.write(output_matrix_file)

        return matrix

    def diagnose_template_coverage(
        self,
        template_source_ascii: str | Path,
        basemap: RasterGrid,
        diagnostics_output_file: str | Path | None = None,
    ) -> CoverageSummary:
        if basemap.geometry != self._target_geometry:
            raise ValueError(
                "Basemap geometry must match mapper target geometry."
            )

        template_grid = AsciiGridReader.read(template_source_ascii)
        transform = self._create_transform_for_geometries(
            source_geometry=template_grid.geometry,
            target_geometry=basemap.geometry,
        )
        diagnostics = MappingDiagnostics(
            transform=transform,
            sampler=BilinearSampler(template_grid),
        )
        diagnostics_grid = diagnostics.classify(
            basemap=basemap,
        )

        if diagnostics_output_file is not None:
            AsciiGridWriter.write(
                raster=diagnostics_grid,
                filename=diagnostics_output_file,
                decimals=0,
            )

        return diagnostics.summarize(
            diagnostics=diagnostics_grid,
        )

    def _create_transform_for_geometries(
        self,
        source_geometry: RasterGeometry,
        target_geometry: RasterGeometry,
    ) -> PiecewiseAffineTransform:
        if not self._use_index_control_points:
            return self._transform

        converted_points = [
            ControlPoint(
                id=point.id,
                source_x=(
                    source_geometry.x_min
                    + point.source_x * source_geometry.cell_size
                ),
                source_y=(
                    source_geometry.y_max
                    - point.source_y * source_geometry.cell_size
                ),
                target_x=(
                    target_geometry.x_min
                    + point.target_x * target_geometry.cell_size
                ),
                target_y=(
                    target_geometry.y_max
                    - point.target_y * target_geometry.cell_size
                ),
                name=point.name,
            )
            for point in self._transform.control_points
        ]

        return PiecewiseAffineTransform(
            control_points=converted_points,
            triangles=list(self._transform.triangle_ids),
        )

    def warp_files(
        self,
        source_files: Iterable[str | Path],
        output_directory: str | Path,
        matrix: MappingMatrix,
        decimals: int = 6,
    ) -> BatchWarpStats:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        matrix_cache: dict[
            tuple[int, int, float, float, float],
            MappingMatrix,
        ] = {}

        processed = 0

        for source_file in source_files:
            source_file = Path(source_file)
            source_grid = AsciiGridReader.read(source_file)
            source_matrix = self._matrix_for_source_geometry(
                matrix=matrix,
                source_geometry=source_grid.geometry,
                matrix_cache=matrix_cache,
            )

            warped = source_matrix.warp(
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
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        files = sorted(source_directory.rglob(pattern))
        matrix_cache: dict[
            tuple[int, int, float, float, float],
            MappingMatrix,
        ] = {}

        processed = 0

        for source_file in files:
            source_grid = AsciiGridReader.read(source_file)
            source_matrix = self._matrix_for_source_geometry(
                matrix=matrix,
                source_geometry=source_grid.geometry,
                matrix_cache=matrix_cache,
            )
            warped = source_matrix.warp(
                sampler=BilinearSampler(source_grid),
            )

            relative_path = source_file.relative_to(source_directory)
            output_path = output_directory / relative_path
            AsciiGridWriter.write(
                raster=warped,
                filename=output_path,
                decimals=decimals,
            )

            processed += 1

        return BatchWarpStats(processed_files=processed)

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
        import numpy as np

        source_directory = Path(source_directory)
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        files = sorted(source_directory.rglob(pattern))

        diagnostics_directory = (
            Path(diagnostics_directory)
            if diagnostics_directory is not None
            else None
        )
        if diagnostics_directory is not None:
            diagnostics_directory.mkdir(parents=True, exist_ok=True)
        matrix_cache: dict[
            tuple[int, int, float, float, float],
            MappingMatrix,
        ] = {}

        processed = 0

        for source_file in files:
            source_grid = AsciiGridReader.read(source_file)
            source_matrix = self._matrix_for_source_geometry(
                matrix=matrix,
                source_geometry=source_grid.geometry,
                matrix_cache=matrix_cache,
            )
            warped = source_matrix.warp(
                sampler=BilinearSampler(source_grid),
            )

            relative_path = source_file.relative_to(source_directory)
            output_path = output_directory / relative_path
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

                diag_path = (
                    diagnostics_directory
                    / relative_path.parent
                    / f"{relative_path.stem}_diag.asc"
                )
                AsciiGridWriter.write(
                    raster=diag_grid,
                    filename=diag_path,
                )

            processed += 1

        return BatchWarpStats(processed_files=processed)

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
        matrix_cache: dict[
            tuple[int, int, float, float, float],
            MappingMatrix,
        ] = {}

        processed = 0

        for source_file in source_files:
            source_file = Path(source_file)
            source_grid = AsciiGridReader.read(source_file)
            source_matrix = self._matrix_for_source_geometry(
                matrix=matrix,
                source_geometry=source_grid.geometry,
                matrix_cache=matrix_cache,
            )

            warped = source_matrix.warp(
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

    def _matrix_for_source_geometry(
        self,
        matrix: MappingMatrix,
        source_geometry: RasterGeometry,
        matrix_cache: dict[tuple[int, int, float, float, float], MappingMatrix],
    ) -> MappingMatrix:
        if matrix.is_compatible_source_geometry(source_geometry):
            return matrix

        if not self._use_index_control_points:
            return matrix

        if (
            matrix.source_geometry.rows != source_geometry.rows
            or matrix.source_geometry.columns != source_geometry.columns
        ):
            return matrix

        geometry_key = (
            source_geometry.rows,
            source_geometry.columns,
            source_geometry.x_min,
            source_geometry.y_min,
            source_geometry.cell_size,
        )

        cached = matrix_cache.get(geometry_key)
        if cached is not None:
            return cached

        matrix_source = matrix.source_geometry
        source_column = (
            (matrix.source_x - matrix_source.x_min)
            / matrix_source.cell_size
        )
        source_row = (
            (matrix_source.y_max - matrix.source_y)
            / matrix_source.cell_size
        )

        remapped_source_x = (
            source_geometry.x_min
            + source_column * source_geometry.cell_size
        )
        remapped_source_y = (
            source_geometry.y_max
            - source_row * source_geometry.cell_size
        )

        remapped_matrix = MappingMatrix(
            source_geometry=source_geometry,
            target_geometry=matrix.target_geometry,
            strategy=matrix.strategy,
            source_x=remapped_source_x,
            source_y=remapped_source_y,
            success=matrix.success,
            triangle_index=matrix.triangle_index,
        )

        matrix_cache[geometry_key] = remapped_matrix
        return remapped_matrix

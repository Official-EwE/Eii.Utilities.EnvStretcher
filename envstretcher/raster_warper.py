from __future__ import annotations

import numpy as np

from envstretcher.bilinear_sampler import BilinearSampler
from envstretcher.mapping_definition import MappingStrategy
from envstretcher.mapping_matrix import MappingMatrix
from envstretcher.piecewise_affine_transform import PiecewiseAffineTransform
from envstretcher.raster_geometry import RasterGeometry
from envstretcher.raster_grid import RasterGrid


class RasterWarper:
    """
    Warp sampled source values onto a target raster geometry.

    The warper creates a new RasterGrid and fills it by:

        target cell centre
            -> target-to-source mapping
            -> source sampling
            -> target value
    """

    def __init__(
        self,
        sampler: BilinearSampler,
        transform: PiecewiseAffineTransform,
    ) -> None:
        self._sampler = sampler
        self._transform = transform

    def warp(
        self,
        target_geometry: RasterGeometry,
        nodata: float = np.nan,
        strategy: MappingStrategy = MappingStrategy.CENTROID,
    ) -> RasterGrid:
        """
        Create and populate a raster with the requested target geometry.
        """

        mapping = self.build_mapping_matrix(
            target_geometry=target_geometry,
            strategy=strategy,
        )

        return self.warp_with_mapping(
            mapping=mapping,
            nodata=nodata,
        )

    def build_mapping_matrix(
        self,
        target_geometry: RasterGeometry,
        strategy: MappingStrategy = MappingStrategy.CENTROID,
    ) -> MappingMatrix:
        return MappingMatrix.build(
            transform=self._transform,
            source_geometry=self._sampler.grid.geometry,
            target_geometry=target_geometry,
            strategy=strategy,
        )

    def warp_with_mapping(
        self,
        mapping: MappingMatrix,
        nodata: float = np.nan,
    ) -> RasterGrid:
        return mapping.warp(
            sampler=self._sampler,
            nodata=nodata,
        )

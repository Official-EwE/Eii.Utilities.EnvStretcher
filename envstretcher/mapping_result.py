from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True, slots=True)
class MappingResult:
    """
    Result of mapping one target coordinate to the source domain.

    A mapping fails when the target coordinate lies outside the convex hull
    of the target control points.
    """

    success: bool

    source_x: float = np.nan
    source_y: float = np.nan

    triangle_index: int = -1

    @classmethod
    def succeeded(
        cls,
        source_x: float,
        source_y: float,
        triangle_index: int,
    ) -> MappingResult:
        return cls(
            success=True,
            source_x=source_x,
            source_y=source_y,
            triangle_index=triangle_index,
        )

    @classmethod
    def failed(cls) -> MappingResult:
        return cls(success=False)
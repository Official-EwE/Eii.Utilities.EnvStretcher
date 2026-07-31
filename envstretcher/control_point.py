from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class ControlPoint:
    """
    A corresponding point in the source and target coordinate systems.
    """

    id: str
    name: str

    source_x: float
    source_y: float

    target_x: float
    target_y: float


    def __init__(
        self,
        id: str,
        source_x: float,
        source_y: float,
        target_x: float,
        target_y: float,
        name: str = "",
    ) -> None:
        self.id = id
        self.source_x = source_x
        self.source_y = source_y
        self.target_x = target_x
        self.target_y = target_y
        self.name = name

    @property
    def source(self) -> tuple[float, float]:
        return self.source_x, self.source_y

    @property
    def target(self) -> tuple[float, float]:
        return self.target_x, self.target_y
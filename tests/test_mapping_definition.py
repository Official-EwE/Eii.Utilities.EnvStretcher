from __future__ import annotations

import pytest

from envstretcher.mapping_definition import MappingDefinition
from envstretcher.mapping_definition import MappingStrategy


def _minimal_mapping_dict() -> dict[str, object]:
    return {
        "points": [
            {"id": "1", "source": [0, 0], "target": [0, 0]},
            {"id": "2", "source": [1, 0], "target": [1, 0]},
            {"id": "3", "source": [0, 1], "target": [0, 1]},
        ]
    }


def test_mapping_definition_defaults_to_centroid_strategy() -> None:
    mapping = MappingDefinition.from_dict(_minimal_mapping_dict())

    assert mapping.mapping_strategy is MappingStrategy.CENTROID


def test_mapping_definition_reads_mapping_strategy() -> None:
    data = _minimal_mapping_dict()
    data["mapping_strategy"] = "max_cell_area"

    mapping = MappingDefinition.from_dict(data)

    assert mapping.mapping_strategy is MappingStrategy.MAX_CELL_AREA


def test_mapping_definition_rejects_invalid_mapping_strategy() -> None:
    data = _minimal_mapping_dict()
    data["mapping_strategy"] = "not-a-strategy"

    with pytest.raises(ValueError, match="Invalid mapping strategy"):
        MappingDefinition.from_dict(data)


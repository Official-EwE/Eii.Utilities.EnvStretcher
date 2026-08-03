from __future__ import annotations

import json
from pathlib import Path

import pytest

from envstretcher.mapping_definition import MappingDefinition
from envstretcher.mapping_definition import MappingStrategy
from envstretcher.piecewise_affine_transform import PiecewiseAffineTransform


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


def test_to_transform_returns_piecewise_affine_transform() -> None:
    mapping = MappingDefinition.from_dict(_minimal_mapping_dict())

    transform = mapping.to_transform()

    assert isinstance(transform, PiecewiseAffineTransform)
    assert len(transform.control_points) == 3


def test_to_transform_uses_delaunay_when_no_triangles() -> None:
    mapping = MappingDefinition.from_dict(_minimal_mapping_dict())

    transform = mapping.to_transform()

    assert len(transform.triangles) == 1


def test_to_transform_uses_explicit_triangles() -> None:
    data = _minimal_mapping_dict()
    data["triangles"] = [["1", "2", "3"]]

    mapping = MappingDefinition.from_dict(data)
    transform = mapping.to_transform()

    assert len(transform.triangles) == 1
    assert set(transform.triangle_ids[0]) == {"1", "2", "3"}


def test_write_round_trips_mapping_definition(
    tmp_path: Path,
) -> None:
    data = {
        "points": [
            {
                "id": "1",
                "name": "North breakwater",
                "source": [0, 0],
                "target": [10, 10],
            },
            {
                "id": "2",
                "source": [1, 0],
                "target": [11, 10],
            },
            {
                "id": "3",
                "source": [0, 1],
                "target": [10, 11],
            },
        ],
        "triangles": [["1", "2", "3"]],
        "mapping_strategy": "max_cell_area",
    }

    mapping = MappingDefinition.from_dict(data)
    output_file = tmp_path / "mapping.json"

    mapping.write(output_file)

    written_data = json.loads(
        output_file.read_text(encoding="utf-8")
    )

    assert written_data == {
        "points": [
            {
                "id": "1",
                "name": "North breakwater",
                "source": [0.0, 0.0],
                "target": [10.0, 10.0],
            },
            {
                "id": "2",
                "source": [1.0, 0.0],
                "target": [11.0, 10.0],
            },
            {
                "id": "3",
                "source": [0.0, 1.0],
                "target": [10.0, 11.0],
            },
        ],
        "triangles": [["1", "2", "3"]],
        "mapping_strategy": "max_cell_area",
    }
    assert MappingDefinition.read(output_file) == mapping

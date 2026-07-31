import numpy as np

from envstretcher.mapping_result import MappingResult


def test_mapping_result_succeeded() -> None:
    result = MappingResult.succeeded(
        source_x=12.5,
        source_y=8.25,
        triangle_index=3,
    )

    assert result.success
    assert result.source_x == 12.5
    assert result.source_y == 8.25
    assert result.triangle_index == 3


def test_mapping_result_failed() -> None:
    result = MappingResult.failed()

    assert not result.success
    assert np.isnan(result.source_x)
    assert np.isnan(result.source_y)
    assert result.triangle_index == -1

import pytest
from dataclasses import FrozenInstanceError


def test_mapping_result_is_immutable() -> None:
    result = MappingResult.failed()

    with pytest.raises(FrozenInstanceError):
        result.success = True
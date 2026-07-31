import pytest

from envstretcher.raster_geometry import RasterGeometry


def test_cell_center() -> None:
    geometry = RasterGeometry(
        rows=3,
        columns=4,
        x_min=10.0,
        y_min=20.0,
        cell_size=2.0,
    )

    assert geometry.cell_center(0, 0) == (11.0, 25.0)
    assert geometry.cell_center(2, 3) == (17.0, 21.0)

@pytest.mark.parametrize(
    ("row", "column"),
    [
        (-1, 0),
        (0, -1),
        (3, 0),
        (0, 4),
    ],
)
def test_cell_center_rejects_invalid_indices(
    row: int,
    column: int,
) -> None:
    geometry = RasterGeometry(
        rows=3,
        columns=4,
        x_min=10.0,
        y_min=20.0,
        cell_size=2.0,
    )

    with pytest.raises(IndexError):
        geometry.cell_center(row, column)
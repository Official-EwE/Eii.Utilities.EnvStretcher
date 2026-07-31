import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class TriangleTransform:
    """
    Affine target-to-source transform for one Delaunay triangle.

    The affine transformation matrix is calculated once during construction.
    Subsequent point transformations therefore require only a matrix
    multiplication.
    """

    def __init__(
        self,
        source_vertices: FloatArray,
        target_vertices: FloatArray,
    ) -> None:
        self._validate_vertices(source_vertices, target_vertices)

        self._matrix = self._build_matrix(
            source_vertices=source_vertices,
            target_vertices=target_vertices,
        )

    @staticmethod
    def _validate_vertices(
        source_vertices: FloatArray,
        target_vertices: FloatArray,
    ) -> None:
        expected_shape = (3, 2)

        if source_vertices.shape != expected_shape:
            raise ValueError(
                "source_vertices must have shape (3, 2), "
                f"but received {source_vertices.shape}."
            )

        if target_vertices.shape != expected_shape:
            raise ValueError(
                "target_vertices must have shape (3, 2), "
                f"but received {target_vertices.shape}."
            )

        if not np.all(np.isfinite(source_vertices)):
            raise ValueError("source_vertices contains non-finite values.")

        if not np.all(np.isfinite(target_vertices)):
            raise ValueError("target_vertices contains non-finite values.")

    @staticmethod
    def _build_matrix(
        source_vertices: FloatArray,
        target_vertices: FloatArray,
    ) -> FloatArray:
        """
        Calculate the 3×2 affine transformation matrix.

        For a target point represented as

            [target_x, target_y, 1]

        the corresponding source point is calculated as

            [target_x, target_y, 1] @ matrix
        """

        target_homogeneous = np.column_stack(
            (
                target_vertices,
                np.ones(3, dtype=np.float64),
            )
        )

        try:
            return np.linalg.solve(
                target_homogeneous,
                source_vertices,
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Cannot construct an affine transform from a "
                "degenerate target triangle."
            ) from exc

    def target_to_source(
        self,
        target_x: float,
        target_y: float,
    ) -> tuple[float, float]:
        """
        Transform one coordinate from the target domain to the source domain.
        """

        target_point = np.array(
            [target_x, target_y, 1.0],
            dtype=np.float64,
        )

        source_point = target_point @ self._matrix

        return (
            float(source_point[0]),
            float(source_point[1]),
        )
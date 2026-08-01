from __future__ import annotations

from pathlib import Path

import shapefile

from envstretcher.piecewise_affine_transform import PiecewiseAffineTransform


class TriangulationShapefileExporter:
    """
    Export the triangulation of a PiecewiseAffineTransform to shapefiles.

    Writes two polygon shapefiles:
    - source: one triangle polygon per affine patch, in source map space.
    - target: one triangle polygon per affine patch, in target map space.

    Each feature carries attributes:
    - triangle_number (int): 1-based index.
    - point_ids      (str): comma-separated IDs of the three vertices.
    - id_a, id_b, id_c (str): individual vertex IDs.
    """

    def __init__(
        self,
        transform: PiecewiseAffineTransform,
    ) -> None:
        self._transform = transform

    def write(
        self,
        source_shapefile: str | Path,
        target_shapefile: str | Path,
    ) -> None:
        """
        Write source and target shapefiles.

        Paths should be given without extension; .shp/.dbf/.shx are
        created automatically by pyshp.
        """
        source_shapefile = Path(source_shapefile)
        target_shapefile = Path(target_shapefile)

        source_shapefile.parent.mkdir(parents=True, exist_ok=True)
        target_shapefile.parent.mkdir(parents=True, exist_ok=True)

        self._write_shapefile(
            path=source_shapefile,
            use_source=True,
        )
        self._write_shapefile(
            path=target_shapefile,
            use_source=False,
        )

    def _write_shapefile(
        self,
        path: Path,
        use_source: bool,
    ) -> None:
        with shapefile.Writer(str(path), shapeType=shapefile.POLYGON) as writer:
            writer.field("tri_number", "N", decimal=0)
            writer.field("point_ids", "C", size=64)
            writer.field("id_a", "C", size=16)
            writer.field("id_b", "C", size=16)
            writer.field("id_c", "C", size=16)

            for number, triangle in enumerate(
                self._transform.triangles,
                start=1,
            ):
                vertices = (
                    triangle.source_vertices
                    if use_source
                    else triangle.target_vertices
                )

                # Closed ring: repeat the first vertex at the end.
                ring = [
                    [float(vertices[0, 0]), float(vertices[0, 1])],
                    [float(vertices[1, 0]), float(vertices[1, 1])],
                    [float(vertices[2, 0]), float(vertices[2, 1])],
                    [float(vertices[0, 0]), float(vertices[0, 1])],
                ]

                writer.poly([ring])
                writer.record(
                    tri_number=number,
                    point_ids=",".join(triangle.point_ids),
                    id_a=triangle.point_ids[0],
                    id_b=triangle.point_ids[1],
                    id_c=triangle.point_ids[2],
                )

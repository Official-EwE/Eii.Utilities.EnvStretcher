import numpy as np

from envstretcher.bilinear_sampler import BilinearSampler
from envstretcher.control_point import ControlPoint
from envstretcher.mapping_definition import MappingStrategy
from envstretcher.piecewise_affine_transform import PiecewiseAffineTransform
from envstretcher.raster_geometry import RasterGeometry
from envstretcher.raster_grid import RasterGrid
from envstretcher.raster_warper import RasterWarper
from envstretcher.io.ascii_reader import AsciiGridReader


def run_synthetic_warp(
    triangles: tuple[tuple[str, str, str], ...] | None = None,
    strategy: MappingStrategy = MappingStrategy.CENTROID,
) -> tuple[RasterGrid, RasterGrid, np.ndarray]:

    # -------------------------------------------------------------------------
    # 1. Create a 10 x 10 source raster
    # -------------------------------------------------------------------------

    source_geometry = RasterGeometry(
        rows=10,
        columns=10,
        x_min=0.0,
        y_min=0.0,
        cell_size=1.0,
    )

    source_values = np.empty(
        shape=(source_geometry.rows, source_geometry.columns),
        dtype=np.float64,
    )

    # Fill the raster with a continuous linear field evaluated at cell centres:
    #
    #     value = 100 + 2x + 3y
    #
    # Bilinear interpolation should reproduce this field exactly.

    for row in range(source_geometry.rows):
        for column in range(source_geometry.columns):

            x, y = source_geometry.cell_center(
                row=row,
                column=column,
            )

            source_values[row, column] = (
                100.0
                + 2.0 * x
                + 3.0 * y
            )

    source_grid = RasterGrid(
        geometry=source_geometry,
        values=source_values,
        nodata=np.nan,
    )


    # -------------------------------------------------------------------------
    # 2. Define the 6 x 6 target geometry
    # -------------------------------------------------------------------------

    target_geometry = RasterGeometry(
        rows=6,
        columns=6,
        x_min=0.0,
        y_min=0.0,
        cell_size=1.0,
    )


    # -------------------------------------------------------------------------
    # 3. Define target-to-source control points
    # -------------------------------------------------------------------------
    #
    # The target domain covers:
    #
    #     x = 0 .. 6
    #     y = 0 .. 6
    #
    # The source domain covers:
    #
    #     x = 0 .. 10
    #     y = 0 .. 10
    #
    # Therefore:
    #
    #     source_x = target_x * 10 / 6
    #     source_y = target_y * 10 / 6

    control_points = [
        ControlPoint(
            id="1",
            source_x=0.0,
            source_y=0.0,
            target_x=0.0,
            target_y=0.0,
        ),
        ControlPoint(
            id="2",
            source_x=10.0,
            source_y=0.0,
            target_x=6.0,
            target_y=0.0,
        ),
        ControlPoint(
            id="3",
            source_x=10.0,
            source_y=10.0,
            target_x=6.0,
            target_y=6.0,
        ),
        ControlPoint(
            id="4",
            source_x=0.0,
            source_y=10.0,
            target_x=0.0,
            target_y=6.0,
        ),
    ]

    transform = PiecewiseAffineTransform(
        control_points=control_points,
        triangles=triangles,
    )


    # -------------------------------------------------------------------------
    # 4. Construct sampler and warper
    # -------------------------------------------------------------------------

    sampler = BilinearSampler(
        grid=source_grid,
    )

    warper = RasterWarper(
        sampler=sampler,
        transform=transform,
    )


    # -------------------------------------------------------------------------
    # 5. Perform the warp
    # -------------------------------------------------------------------------

    target_grid = warper.warp(
        target_geometry=target_geometry,
        nodata=np.nan,
        strategy=strategy,
    )


    # -------------------------------------------------------------------------
    # 6. Calculate the mathematically expected result
    # -------------------------------------------------------------------------

    expected_values = np.empty(
        shape=(target_geometry.rows, target_geometry.columns),
        dtype=np.float64,
    )

    scale = 10.0 / 6.0

    for row in range(target_geometry.rows):
        for column in range(target_geometry.columns):

            target_x, target_y = target_geometry.cell_center(
                row=row,
                column=column,
            )

            expected_source_x = target_x * scale
            expected_source_y = target_y * scale

            expected_values[row, column] = (
                100.0
                + 2.0 * expected_source_x
                + 3.0 * expected_source_y
            )


    # -------------------------------------------------------------------------
    # 7. Verify the result
    # -------------------------------------------------------------------------

    return source_grid, target_grid, expected_values



def run_batch_example() -> None:
    """
    Example: two-phase batch workflow for EwE monthly files.

    Phase 1: Build mapping matrix once from a template source.
    Phase 2: Apply matrix to all monthly files.
    """

    from envstretcher.batch_ascii_mapper import BatchAsciiMapper
    from envstretcher.mapping_definition import MappingDefinition
    from envstretcher.mapping_matrix import MappingMatrix
    from diagnostics.triangulation_shapefile_exporter import TriangulationShapefileExporter

    definition = MappingDefinition.read("Resources/nwmed_to_anchovybay_mapping.json")
    transform = definition.to_transform()
    basemap = AsciiGridReader.read("Resources/anchovybay.asc")
    target_geometry = basemap.geometry

    # phase 1
    mapper = BatchAsciiMapper(
        transform,
        target_geometry,
        definition.mapping_strategy,
        use_index_control_points=True,
    )
    coverage = mapper.diagnose_template_coverage(
        template_source_ascii="Resources/nwmed.asc",
        basemap=basemap,
        diagnostics_output_file="D:/nwmed_phase1_coverage.asc",
    )
    print(
        "Phase 1 coverage: "
        f"{coverage.mapped_cells}/{coverage.water_cells} water cells mapped, "
        f"outside={coverage.outside_mapping_cells}, "
        f"source_nodata={coverage.source_nodata_cells}"
    )

    # Export triangulation shapefiles in world (lon/lat) coordinates.
    # build_world_space_transform converts grid-index control points to the
    # actual map coordinates of the source and target rasters.
    source_grid = AsciiGridReader.read("Resources/nwmed.asc")
    world_transform = mapper.build_world_space_transform(
        source_geometry=source_grid.geometry,
        target_geometry=target_geometry,
    )
    exporter = TriangulationShapefileExporter(world_transform)
    exporter.write(
        source_shapefile="D:/nwmed_triangulation_source",
        target_shapefile="D:/nwmed_triangulation_target",
    )
    print("Triangulation shapefiles written.")

    matrix = mapper.build_mapping_matrix("Resources/nwmed.asc", "D:/nwmed_to_anchovybay_mapping.npz")

    # phase 2
    matrix = MappingMatrix.read("D:/nwmed_to_anchovybay_mapping.npz")
    mapper.warp_glob("D:/Projects/2026/ActNow/WP3/NWMed model (FutureMares)/GSA0607EcospaceActNow/Drivers/drivers_annual", "*.asc", "output_monthly", matrix)

    mapper = BatchAsciiMapper(
        transform=transform,
        target_geometry=target_geometry,
        strategy=MappingStrategy.MAX_CELL_AREA,
        use_index_control_points=True,
    )

    print(
        "Batch workflow: phase 1 (matrix build) + "
        "phase 2 (batch warp) ready to use."
    )



def main():
    run_batch_example()

if __name__ == "__main__":
    main()
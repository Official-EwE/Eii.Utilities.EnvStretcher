# Eii.Utilities.EnvStretcher

`EnvStretcher` maps values from a source raster to a target raster using
piecewise-affine geometry and bilinear sampling.

## What is in this repository

- `envstretcher/`: core mapping, raster geometry, and sampling primitives.
  Includes a first-class `Triangle` type that encapsulates triangle-level
  geometry checks and target-to-source mapping.
  Includes `MappingMatrix` for cached mappings and `BatchAsciiMapper` for
  two-phase ASCII processing.
- `diagnostics/`: mapping-coverage classification and summary helpers.
- `io/`: ESRI ASCII grid reader/writer.
- `tests/`: pytest suite for geometry, mapping, and diagnostics behavior.
- `main.py`: synthetic end-to-end example.

## Running tests

Use the project virtual environment:

```powershell
.\.venv\Scripts\python -m pytest -q
```

## Running the demo

```powershell
.\.venv\Scripts\python .\main.py
```

## Mapping flow

1. Build a `PiecewiseAffineTransform` from control points (and optional triangles).
2. For each target cell centre, map target coordinates to source coordinates.
3. Sample source values with `BilinearSampler`.
4. Write values into a new `RasterGrid`.
5. Optionally classify coverage with `MappingDiagnostics`.

## Two-phase batch workflow

1. Phase 1 (calibration):
   - choose control points/triangles and mapping strategy;
   - precompute and save a `MappingMatrix` once.
2. Phase 2 (production):
   - load the matrix;
   - apply it to each monthly ESRI ASCII source raster with matching geometry.

Example:

```python
from io.ascii_reader import AsciiGridReader
from envstretcher.batch_ascii_mapper import BatchAsciiMapper
from envstretcher.control_point import ControlPoint
from envstretcher.mapping_definition import MappingStrategy
from envstretcher.mapping_matrix import MappingMatrix
from envstretcher.piecewise_affine_transform import PiecewiseAffineTransform
from envstretcher.raster_geometry import RasterGeometry

control_points = [
    ControlPoint(id="1", source_x=0.0, source_y=0.0, target_x=0.0, target_y=0.0),
    ControlPoint(id="2", source_x=10.0, source_y=0.0, target_x=6.0, target_y=0.0),
    ControlPoint(id="3", source_x=10.0, source_y=10.0, target_x=6.0, target_y=6.0),
    ControlPoint(id="4", source_x=0.0, source_y=10.0, target_x=0.0, target_y=6.0),
]
transform = PiecewiseAffineTransform(control_points=control_points)

target_geometry = RasterGeometry(
    rows=6,
    columns=6,
    x_min=0.0,
    y_min=0.0,
    cell_size=1.0,
)

# phase 1
mapper = BatchAsciiMapper(transform, target_geometry, MappingStrategy.MAX_CELL_AREA)
matrix = mapper.build_mapping_matrix("month_01.asc", "gol_to_ewe_mapping.npz")

# phase 2
matrix = MappingMatrix.read("gol_to_ewe_mapping.npz")
mapper.warp_glob("input_monthly", "*.asc", "output_monthly", matrix)

# Optional: include diagnostics per file
basemap = AsciiGridReader.read("basemap.asc")
mapper.warp_glob_with_diagnostics(
    "input_monthly", "*.asc",
    "output_monthly", matrix,
    basemap,
    diagnostics_directory="output_diagnostics"
)
```

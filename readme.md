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
- `envstretcher/io/`: ESRI ASCII grid reader/writer.
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
4. Write values into a new `RasterGrid` (source nodata and non-positive
   source cells are excluded from bilinear interpolation and remaining
   weights are renormalized).
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
from envstretcher.io.ascii_reader import AsciiGridReader
from envstretcher.batch_ascii_mapper import BatchAsciiMapper
from envstretcher.mapping_definition import MappingDefinition
from envstretcher.mapping_matrix import MappingMatrix

definition = MappingDefinition.read("nwmed_to_anchovybay_mapping.json")
transform = definition.to_transform()
basemap = AsciiGridReader.read("anchovybay.asc")
target_geometry = basemap.geometry

definition = MappingDefinition.read("nwmed_to_anchovybay_mapping.json")
transform = definition.to_transform()
basemap = AsciiGridReader.read("anchovybay.asc")
target_geometry = basemap.geometry

# phase 1
mapper = BatchAsciiMapper(
    transform,
    target_geometry,
    definition.mapping_strategy,
    use_index_control_points=True,  # EwE control points are grid-index based
)
coverage = mapper.diagnose_template_coverage(
    template_source_ascii="nwmed.asc",
    basemap=basemap,
    diagnostics_output_file="nwmed_phase1_coverage.asc",
)
print(coverage.all_water_cells_covered)
matrix = mapper.build_mapping_matrix("nwmed.asc", "nwmed_to_anchovybay_mapping.npz")

# phase 2
matrix = MappingMatrix.read("nwmed_to_anchovybay_mapping.npz")
mapper.warp_glob("input_monthly", "*.asc", "output_monthly", matrix)

# Optional: include diagnostics per file
mapper.warp_glob_with_diagnostics(
    "input_monthly", "*.asc",
    "output_monthly", matrix,
    basemap,
    diagnostics_directory="output_diagnostics"
)
```
## Funding acknowledgement
Eii.EnvStretcher is developed with the support of the  European Union under the Horizon Europe Program, Grant Agreement No. 101157456 (SURIMI).
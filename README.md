# OpenArmX Painting Control System

## STEP CAD product import

The product-loading screen accepts STEP files (`.step`, `.stp`). STEP is read as an Open CASCADE B-Rep. Exact CAD topology and geometric properties are extracted before a tessellated display mesh is created for the existing PyVista viewer.

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The STEP reader uses `cadquery-ocp` (Open CASCADE Python bindings). The STEP translation target is millimetres. Confirm the originating CAD unit for critical manufacturing measurements.

The first-screen geometry report includes:
- CAD Face, Edge, Vertex, Shell and Solid counts
- Bounding-box minimum, maximum, dimensions and diagonal
- Exact B-Rep surface area, volume, surface centroid and volume centroid
- Total edge length and per-face surface type, area and centroid
- Display mesh vertex/triangle counts and tessellation tolerances
- A mapping from every display triangle to its original CAD Face ID

The full JSON report is saved under `results/<product>/stage1/step_geometry_report.json`.

## Run

```bash
bash run.sh
```

## Existing dependencies

PySide6, PyVista, PyVistaQt, Trimesh, NumPy and SciPy remain the application and geometry dependencies.


## Manual reference pose capture

The manual pose viewer uses a fixed **World XYZ** work-cell frame whose origin is the chain entry point: +X follows chain travel toward screen right, +Y points inward into the work area, and +Z points vertically downward. A configurable wireframe box and bottom-plane grid provide spatial context. Box dimensions are visualization bounds entered by the operator, not measured work-cell dimensions.

The product frame is the STEP **CAD frame**. Its rotation pivot is always the STEP file's existing CAD origin (0, 0, 0); no hook pivot is requested or inferred. Drag the product to rotate it about that CAD origin. Set the CAD origin's world position with the World X/Y/Z fields. The live readout reports the CAD origin position in World XYZ and its roll-pitch-yaw. **CONFIRM & SAVE POSE** appends a reference pose to:

    results/<product>/stage1/reference_poses.json

The saved record includes the World XYZ definition, CAD-origin rotation pivot, CAD-origin world position, RPY, rotation matrix, homogeneous transform, configurable display-box dimensions, source/display units, and UTC timestamp. This is an operator-defined reference pose, not a depth-camera measurement. Camera and robot frames will be added after their mounting and calibration transforms are known; unknown frames are not drawn at invented positions.

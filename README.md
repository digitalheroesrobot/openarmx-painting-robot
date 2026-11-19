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

The manual pose viewer displays two coordinate frames: **World NED**, fixed at the chain-entry point (+X along the chain, +Y lateral, +Z down), and **Product CAD**, attached to the CAD origin and moving with the product. The hook rotation pivot is shown as a magenta point, not as a separate coordinate frame. The live readout reports the Product CAD origin in World NED coordinates and its roll-pitch-yaw. Drag the product actor to adjust its pose about the configured hook pivot. **CONFIRM & SAVE POSE** appends a reference pose to:

    results/<product>/stage1/reference_poses.json

The saved record includes the World NED definition, hook pivot, Product CAD origin position, RPY, rotation matrix, homogeneous transform, source/display units, and UTC timestamp. This is an operator-defined reference pose, not a depth-camera measurement. Camera and robot frames will be added after their mounting and calibration transforms are known; unknown frames are not drawn at invented positions.

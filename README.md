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

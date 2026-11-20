#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Verify the real STEP -> Stage 1 -> Stage 2 -> Stage 3 pipeline.

Run from the repository root:
    PYTHONPATH=. .venv/bin/python scripts/verify_step_pipeline.py /path/to/model.step

The integration run writes its artifacts to:
    <output-root>/<part-name>/<timestamp>/

This is deliberately separate from the GUI's normal results directory so the
verification does not overwrite an operator's existing Stage 1/2/3 results.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from core.result_manager import ResultManager
from core.step_loader import load_step
from stages.cad_surface_analysis import analyze_step_faces
from stages.stage1_geometry import execute_stage1
from stages.stage2_surface_identification import execute_stage2
from stages.stage3_spray_trajectory import execute_stage3


EXACT_BREP_AUTHORITY = "EXACT_STEP_BREP_SHARED_EDGES"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def verify_step_pipeline(
    step_path: Path,
    output_root: Path,
    standoff_mm: float,
    sampling_spacing_mm: float,
    spray_angle_deg: float,
) -> dict:
    step_path = step_path.expanduser().resolve()
    _require(step_path.is_file(), f"STEP file does not exist: {step_path}")
    _require(
        step_path.suffix.lower() in {".step", ".stp"},
        f"Expected a .step or .stp file, got: {step_path.name}",
    )

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_directory = (
        output_root.expanduser().resolve()
        / step_path.stem
        / run_timestamp
    )
    run_directory.mkdir(parents=True, exist_ok=False)

    # ResultManager's output is isolated from the application's normal results.
    result_manager = ResultManager(run_directory)
    checks: dict[str, str] = {}
    report: dict = {
        "status": "RUNNING",
        "source_step_file": str(step_path),
        "run_directory": str(run_directory),
        "timestamp_utc": run_timestamp,
        "parameters": {
            "coordinate_unit": "mm",
            "standoff_mm": float(standoff_mm),
            "sampling_spacing_mm": float(sampling_spacing_mm),
            "spray_angle_deg": float(spray_angle_deg),
        },
        "checks": checks,
    }
    report_path = run_directory / "integration_report.json"

    try:
        print("=" * 76)
        print("REAL STEP -> STAGE 1 -> STAGE 2 -> STAGE 3 INTEGRATION")
        print("=" * 76)
        print(f"STEP: {step_path}")
        print(f"OUTPUT: {run_directory}")

        # ------------------------------------------------------------
        # 1. Load the STEP B-Rep and inspect preserved topology metadata.
        # ------------------------------------------------------------
        mesh = load_step(step_path)
        step_report = mesh.metadata.get("step_geometry_report")
        triangle_face_ids_raw = mesh.metadata.get("triangle_face_ids")

        _require(isinstance(step_report, dict), "STEP geometry report is missing from mesh.metadata.")
        _require(
            isinstance(triangle_face_ids_raw, (list, tuple, np.ndarray)),
            "triangle_face_ids is missing from STEP mesh metadata.",
        )

        triangle_face_ids = np.asarray(triangle_face_ids_raw, dtype=np.int64)
        step_faces = step_report.get("faces", [])
        topology = step_report.get("topology", {})
        face_edge_adjacency = (
            topology.get("face_edge_adjacency", {})
            if isinstance(topology, dict) else {}
        )
        face_adjacency = (
            face_edge_adjacency.get("face_adjacency", {})
            if isinstance(face_edge_adjacency, dict) else {}
        )

        _require(len(mesh.faces) > 0, "STEP tessellation contains no triangles.")
        _require(len(step_faces) > 0, "STEP B-Rep report contains no Faces.")
        _require(
            triangle_face_ids.shape == (len(mesh.faces),),
            "triangle_face_ids must contain exactly one STEP Face ID per mesh triangle.",
        )
        _require(
            isinstance(face_adjacency, dict) and bool(face_adjacency),
            "Exact STEP B-Rep face adjacency is missing or empty.",
        )

        known_step_face_ids = {
            int(record["face_id"])
            for record in step_faces
            if isinstance(record, dict) and "face_id" in record
        }
        _require(bool(known_step_face_ids), "No valid original STEP Face IDs were found.")
        mapped_step_face_ids = set(int(value) for value in np.unique(triangle_face_ids))
        unknown_mapped_ids = mapped_step_face_ids - known_step_face_ids
        _require(
            not unknown_mapped_ids,
            f"Display triangles reference unknown STEP Face IDs: {sorted(unknown_mapped_ids)[:20]}",
        )

        unit = str(mesh.metadata.get("source_unit", "")).lower()
        _require(unit == "mm", f"STEP loader is expected to normalize to mm; got {unit!r}.")

        checks["step_import"] = "PASS"
        checks["triangle_to_step_face_mapping"] = "PASS"
        checks["exact_step_brep_adjacency_available"] = "PASS"
        report["step_geometry"] = {
            "face_count": len(step_faces),
            "edge_count": int(topology.get("edges", 0)) if isinstance(topology, dict) else None,
            "display_triangle_count": int(len(mesh.faces)),
            "mapped_step_face_count": len(mapped_step_face_ids),
            "adjacency_pair_count": int(
                sum(len(values) for values in face_adjacency.values()) // 2
            ),
            "coordinate_unit": unit,
            "source_format": mesh.metadata.get("source_format"),
        }
        print(
            "[PASS] STEP import | "
            f"{len(step_faces)} B-Rep Faces | {len(mesh.faces)} display triangles"
        )
        print("[PASS] Every display triangle maps to a valid original STEP Face ID")
        print("[PASS] Exact STEP B-Rep face adjacency is present")

        original_vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()

        # ------------------------------------------------------------
        # 2. Run the independent STEP Face analysis.
        # ------------------------------------------------------------
        cad_analysis = analyze_step_faces(step_report, mesh)
        cad_authority = cad_analysis["report"]["analysis"].get(
            "face_adjacency_source"
        )
        _require(
            cad_authority == EXACT_BREP_AUTHORITY,
            f"CAD face analysis used {cad_authority!r}, not exact STEP B-Rep adjacency.",
        )
        checks["step_face_analysis"] = "PASS"
        report["step_face_analysis"] = cad_analysis["report"]["summary"]
        print(f"[PASS] STEP Face analysis authority: {cad_authority}")

        # ------------------------------------------------------------
        # 3. Run Stage 1 on the SAME STEP-derived mesh.
        # ------------------------------------------------------------
        stage1_result = execute_stage1(
            mesh=mesh,
            stl_path=step_path,
            result_manager=result_manager,
            source_unit=None,
        )
        stage1_report = stage1_result["report"]
        stage1_patch_labels = np.asarray(stage1_result["patch_labels"])

        _require(
            stage1_patch_labels.shape == (len(mesh.faces),),
            "Stage 1 patch_labels do not match the STEP display triangle count.",
        )
        _require(
            str(stage1_report.get("unit_configuration", {}).get(
                "source_stl_coordinate_unit", ""
            )).lower() == "mm",
            "Stage 1 did not preserve the STEP mesh's millimeter coordinate unit.",
        )
        checks["stage1_geometry_analysis"] = "PASS"
        report["stage1"] = {
            "patch_count": int(len(np.unique(stage1_patch_labels))),
            "patch_label_count": int(len(stage1_patch_labels)),
            "coordinate_unit": stage1_report.get(
                "unit_configuration", {}
            ).get("source_stl_coordinate_unit"),
            "report_path": str(stage1_result["json_path"]),
        }
        print(
            "[PASS] Stage 1 | "
            f"{report['stage1']['patch_count']} patches | "
            f"{len(stage1_patch_labels)} triangle labels"
        )

        # ------------------------------------------------------------
        # 4. Run Stage 2 using the actual Stage 1 result.
        # ------------------------------------------------------------
        stage2_result = execute_stage2(
            mesh=mesh,
            stl_path=step_path,
            result_manager=result_manager,
            stage1_result=stage1_result,
            coordinate_unit="mm",
        )
        stage2_labels = np.asarray(stage2_result["face_status_labels"])
        stage2_patch_labels = np.asarray(stage2_result["patch_labels"])

        _require(
            stage2_labels.shape == (len(mesh.faces),),
            "Stage 2 face_status_labels do not match the STEP display triangle count.",
        )
        _require(
            stage2_patch_labels.shape == (len(mesh.faces),),
            "Stage 2 patch_labels do not match the STEP display triangle count.",
        )

        candidate_count = int(np.count_nonzero(stage2_labels == 1))
        review_count = int(np.count_nonzero(stage2_labels == 2))
        excluded_count = int(np.count_nonzero(stage2_labels == 0))
        invalid_count = int(np.count_nonzero(stage2_labels == 3))
        _require(
            candidate_count > 0,
            "Stage 2 produced zero PAINTABLE_CANDIDATE triangles. "
            "Review Stage 2 classification and the STEP model before Stage 3.",
        )

        checks["stage2_surface_identification"] = "PASS"
        report["stage2"] = {
            "paintable_candidate_triangles": candidate_count,
            "review_required_triangles": review_count,
            "excluded_triangles": excluded_count,
            "invalid_triangles": invalid_count,
            "report_path": str(stage2_result["json_path"]),
            "labels_path": str(stage2_result["labels_path"]),
        }
        print(
            "[PASS] Stage 2 | "
            f"candidate={candidate_count} | review={review_count} | "
            f"excluded={excluded_count} | invalid={invalid_count}"
        )

        # ------------------------------------------------------------
        # 5. Run Stage 3 with Stage 2 labels and Stage 1 report.
        # ------------------------------------------------------------
        stage3_result = execute_stage3(
            mesh=mesh,
            stl_path=step_path,
            result_manager=result_manager,
            stage2_result=stage2_result,
            stage1_report=stage1_report,
            coordinate_unit="mm",
            standoff_mm=standoff_mm,
            sampling_spacing_mm=sampling_spacing_mm,
            spray_angle_deg=spray_angle_deg,
        )
        trajectory = stage3_result["trajectory"]
        sample_count = int(stage3_result["sample_count"])
        _require(sample_count > 0, "Stage 3 generated zero trajectory samples.")

        trajectory_face_indices = np.asarray(
            trajectory["face_indices"], dtype=np.int64
        )
        _require(
            trajectory_face_indices.shape == (sample_count,),
            "Stage 3 face_indices length does not match its sample count.",
        )
        _require(
            np.all(trajectory_face_indices >= 0)
            and np.all(trajectory_face_indices < len(mesh.faces)),
            "Stage 3 contains an out-of-range display triangle index.",
        )

        trajectory_step_face_ids = triangle_face_ids[trajectory_face_indices]
        _require(
            set(int(value) for value in np.unique(trajectory_step_face_ids))
            <= known_step_face_ids,
            "Stage 3 trajectory contains a sample that cannot be mapped to an original STEP Face.",
        )

        boundary_diagnostics = trajectory.get("surface_boundary_splits", {})
        stage3_authority = boundary_diagnostics.get("adjacency_authority")
        _require(
            stage3_authority == EXACT_BREP_AUTHORITY,
            "Stage 3 did not use exact STEP B-Rep adjacency. "
            f"Reported authority: {stage3_authority!r}",
        )
        _require(
            boundary_diagnostics.get("samples_preserved") is True,
            "Stage 3 surface-boundary splitting did not confirm sample preservation.",
        )

        checks["stage3_trajectory_generation"] = "PASS"
        checks["stage3_exact_step_adjacency"] = "PASS"
        checks["trajectory_samples_map_to_step_faces"] = "PASS"
        report["stage3"] = {
            "sample_count": sample_count,
            "segment_count": int(len(trajectory.get("segments", []))),
            "mapped_step_face_count": int(len(np.unique(trajectory_step_face_ids))),
            "surface_boundary_splits": boundary_diagnostics,
            "trajectory_validation": trajectory.get("trajectory_validation", {}),
            "report_path": str(stage3_result["json_path"]),
            "trajectory_array_path": str(stage3_result["array_path"]),
        }
        print(
            "[PASS] Stage 3 | "
            f"{sample_count} samples | "
            f"{len(trajectory.get('segments', []))} strokes"
        )
        print(f"[PASS] Stage 3 adjacency authority: {stage3_authority}")
        print("[PASS] Every trajectory sample maps to an original STEP Face ID")

        # ------------------------------------------------------------
        # 6. Ensure the original STEP display mesh was not modified.
        # ------------------------------------------------------------
        _require(
            np.array_equal(
                np.asarray(mesh.vertices, dtype=np.float64),
                original_vertices,
            ),
            "The source STEP display mesh vertices were modified by the pipeline.",
        )
        checks["source_mesh_unchanged"] = "PASS"

        report["status"] = "PASS"
        report["checks_passed"] = len(checks)
        report["checks_total"] = len(checks)
        _json_write(report_path, report)

        print("=" * 76)
        print(f"INTEGRATION RESULT: PASS ({len(checks)}/{len(checks)} checks)")
        print(f"REPORT: {report_path}")
        print("=" * 76)
        return report

    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        report["checks_passed"] = sum(value == "PASS" for value in checks.values())
        report["checks_total"] = None
        _json_write(report_path, report)
        print("=" * 76, file=sys.stderr)
        print(f"INTEGRATION RESULT: FAIL - {exc}", file=sys.stderr)
        print(f"PARTIAL REPORT: {report_path}", file=sys.stderr)
        print("=" * 76, file=sys.stderr)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a real STEP -> Stage 1 -> Stage 2 -> Stage 3 integration test."
    )
    parser.add_argument(
        "step_file",
        type=Path,
        help="Path to the actual .step or .stp product model.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("integration_test_output"),
        help="Separate output root; defaults to ./integration_test_output.",
    )
    parser.add_argument(
        "--standoff-mm",
        type=float,
        default=200.0,
        help="Nozzle standoff in millimeters (default: 200).",
    )
    parser.add_argument(
        "--sampling-spacing-mm",
        type=float,
        default=50.0,
        help="Surface sampling spacing in millimeters (default: 50).",
    )
    parser.add_argument(
        "--spray-angle-deg",
        type=float,
        default=74.0,
        help="Spray cone/fan angle in degrees (default: 74).",
    )
    args = parser.parse_args()

    try:
        verify_step_pipeline(
            step_path=args.step_file,
            output_root=args.output_root,
            standoff_mm=args.standoff_mm,
            sampling_spacing_mm=args.sampling_spacing_mm,
            spray_angle_deg=args.spray_angle_deg,
        )
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

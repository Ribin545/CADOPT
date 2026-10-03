from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def residual_stats(values: np.ndarray) -> Dict[str, float]:
    """Return mean/max/rms statistics for a residual vector."""
    if values.size == 0:
        return {"mean": 0.0, "max": 0.0, "rms": 0.0}
    vv = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(vv)),
        "max": float(np.max(vv)),
        "rms": float(np.sqrt(np.mean(vv * vv))),
    }


def format_metrics_table(run_metrics: Dict) -> str:
    """Create a compact human-readable table for one pipeline run."""
    phases = run_metrics.get("phases", {})

    header = "Phase | Time (s) | Status | Polycount / Notes"
    sep = "-" * len(header)
    lines: List[str] = [header, sep]

    # Use specified order for standard phases, but allow dynamic ones
    standard_order = ["phase1", "phase2", "phase3", "phase4", "phase5"]
    other_phases = sorted([k for k in phases.keys() if k not in standard_order])
    all_names = standard_order + other_phases

    for phase_name in all_names:
        if phase_name not in phases and phase_name.startswith("phase"):
            # Only show standard phases as 'skip' if they are missing
            dt = 0.0
            ok = "skip"
            notes = ""
            lines.append(f"{phase_name:5s} | {dt:8.4f} | {ok:6s} | {notes}")
        elif phase_name in phases:
            p = phases[phase_name]
            dt = float(p.get("time_s", 0.0))
            ok = "ok" if p.get("ok", False) else "fail"
            
            # Polycount info for notes if not provided
            notes = p.get("notes", "")
            if not notes:
                if "n_verts" in p:
                    notes = f"verts={p['n_verts']}"
                    if "n_tris" in p:
                        notes += f", tris={p['n_tris']}"
                elif "removed_faces" in p:
                    notes = f"removed={p['removed_faces']}"
                elif phase_name == "qa_validation":
                    report = p.get("report", {})
                    errors = report.get("errors", {})
                    if errors:
                        notes = f"fails={list(errors.keys())}"
                    else:
                        notes = "Passed"
            
            lines.append(f"{phase_name:12s} | {dt:8.4f} | {ok:6s} | {notes}")

    lines.append(sep)
    lines.append(f"total | {float(run_metrics.get('total_time_s', 0.0)):.4f} s")
    return "\n".join(lines)


def flatten_runs_for_csv(runs: Iterable[Dict]) -> List[Dict[str, object]]:
    """Flatten nested run metrics for simple CSV export."""
    rows: List[Dict[str, object]] = []
    for i, run in enumerate(runs):
        phases = run.get("phases", {})
        row: Dict[str, object] = {
            "run": i,
            "step_path": run.get("step_path", ""),
            "total_time_s": run.get("total_time_s", 0.0),
        }
        for phase_name, pdata in phases.items():
            row[f"{phase_name}_ok"] = pdata.get("ok", False)
            row[f"{phase_name}_time_s"] = pdata.get("time_s", 0.0)
            if "n_verts" in pdata:
                row[f"{phase_name}_n_verts"] = pdata.get("n_verts", 0)
            if "n_tris" in pdata:
                row[f"{phase_name}_n_tris"] = pdata.get("n_tris", 0)
            if "n_faces" in pdata:
                row[f"{phase_name}_n_faces"] = pdata.get("n_faces", 0)
            if "face_arity" in pdata:
                row[f"{phase_name}_face_arity"] = pdata.get("face_arity", 0)
            if "n_quads" in pdata:
                row[f"{phase_name}_n_quads"] = pdata.get("n_quads", 0)
            if "n_tris_out" in pdata:
                row[f"{phase_name}_n_tris_out"] = pdata.get("n_tris_out", 0)
            if "proj_err_mean" in pdata:
                row[f"{phase_name}_proj_err_mean"] = pdata.get("proj_err_mean", 0.0)
                row[f"{phase_name}_proj_err_max"] = pdata.get("proj_err_max", 0.0)
                row[f"{phase_name}_proj_err_rms"] = pdata.get("proj_err_rms", 0.0)
            if "mesh_n_components" in pdata:
                row[f"{phase_name}_mesh_n_components"] = pdata.get("mesh_n_components", 0)
            if "mesh_boundary_edges" in pdata:
                row[f"{phase_name}_mesh_boundary_edges"] = pdata.get("mesh_boundary_edges", 0)
            if "mesh_n_degenerate_tris" in pdata:
                row[f"{phase_name}_mesh_n_degenerate_tris"] = pdata.get("mesh_n_degenerate_tris", 0)
            if "mesh_largest_component_n_faces" in pdata:
                row[f"{phase_name}_mesh_largest_component_n_faces"] = pdata.get(
                    "mesh_largest_component_n_faces", 0
                )
            if "mesh_edge_len_mean" in pdata:
                row[f"{phase_name}_mesh_edge_len_mean"] = pdata.get("mesh_edge_len_mean", 0.0)
            if "mesh_edge_len_min" in pdata:
                row[f"{phase_name}_mesh_edge_len_min"] = pdata.get("mesh_edge_len_min", 0.0)
            if "mesh_edge_len_max" in pdata:
                row[f"{phase_name}_mesh_edge_len_max"] = pdata.get("mesh_edge_len_max", 0.0)
            if "mesh_tri_aspect_mean" in pdata:
                row[f"{phase_name}_mesh_tri_aspect_mean"] = pdata.get("mesh_tri_aspect_mean", 0.0)
            if "mesh_tri_aspect_p95" in pdata:
                row[f"{phase_name}_mesh_tri_aspect_p95"] = pdata.get("mesh_tri_aspect_p95", 0.0)
            if "mesh_tri_aspect_max" in pdata:
                row[f"{phase_name}_mesh_tri_aspect_max"] = pdata.get("mesh_tri_aspect_max", 0.0)
            if "mesh_valence_mean" in pdata:
                row[f"{phase_name}_mesh_valence_mean"] = pdata.get("mesh_valence_mean", 0.0)
            if "mesh_valence_max" in pdata:
                row[f"{phase_name}_mesh_valence_max"] = pdata.get("mesh_valence_max", 0.0)
            if "shading_harmonization_count" in pdata:
                row[f"{phase_name}_shading_harmonization_count"] = pdata.get("shading_harmonization_count", 0)
            if "shading_max_jump_deg" in pdata:
                row[f"{phase_name}_shading_max_jump_deg"] = pdata.get("shading_max_jump_deg", 0.0)
            if "shading_edge_snapping_count" in pdata:
                row[f"{phase_name}_shading_edge_snapping_count"] = pdata.get("shading_edge_snapping_count", 0)
            if "shading_antipodal_flip_count" in pdata:
                row[f"{phase_name}_shading_antipodal_flip_count"] = pdata.get("shading_antipodal_flip_count", 0)
            if "shading_chamfer_edge_collapse_count" in pdata:
                row[f"{phase_name}_shading_chamfer_edge_collapse_count"] = pdata.get("shading_chamfer_edge_collapse_count", 0)
            if "shading_smooth_weld_merge_count" in pdata:
                row[f"{phase_name}_shading_smooth_weld_merge_count"] = pdata.get("shading_smooth_weld_merge_count", 0)
            if "shading_normal_relax_iters" in pdata:
                row[f"{phase_name}_shading_normal_relax_iters"] = pdata.get("shading_normal_relax_iters", 0)
            if "shading_normal_relax_strength" in pdata:
                row[f"{phase_name}_shading_normal_relax_strength"] = pdata.get("shading_normal_relax_strength", 0.0)
        rows.append(row)
    return rows

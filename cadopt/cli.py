from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

def _write_obj(path: Path, V, F, VN=None, VT=None) -> None:
    """Write triangle/quad mesh to Wavefront OBJ with optional normals and UVs."""
    lines = []
    for p in V:
        lines.append(f"v {float(p[0]):.12g} {float(p[1]):.12g} {float(p[2]):.12g}")
    
    has_uv = VT is not None and len(VT) == len(V)
    if has_uv:
        for uv in VT:
            lines.append(f"vt {float(uv[0]):.12g} {float(uv[1]):.12g}")

    has_normals = VN is not None and len(VN) == len(V)
    if has_normals:
        for n in VN:
            lines.append(f"vn {float(n[0]):.12g} {float(n[1]):.12g} {float(n[2]):.12g}")

    for f in F:
        idx = [int(x) + 1 for x in f.tolist()]
        if has_uv and has_normals:
            lines.append("f " + " ".join(f"{i}/{i}/{i}" for i in idx))
        elif has_uv:
            lines.append("f " + " ".join(f"{i}/{i}" for i in idx))
        elif has_normals:
            lines.append("f " + " ".join(f"{i}//{i}" for i in idx))
        else:
            lines.append("f " + " ".join(str(i) for i in idx))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _write_blender_script(path: Path, obj_filename: str, V, F, feature_lines=None) -> None:
    """Write a Python script for Blender to import OBJ and mark sharp/crease edges."""
    script = [
        "import bpy",
        "import bmesh",
        "import os",
        "from mathutils import Vector",
        "",
        f"obj_path = os.path.join(os.path.dirname(__file__), '{obj_filename}')",
        "if os.path.exists(obj_path):",
        "    bpy.ops.wm.obj_import(filepath=obj_path)",
        "    obj = bpy.context.selected_objects[0]",
        "    bpy.context.view_layer.objects.active = obj",
        "",
        "    if hasattr(obj.data, 'use_auto_smooth'):",
        "        obj.data.use_auto_smooth = True",
        "    ",
        "    bm = bmesh.new()",
        "    bm.from_mesh(obj.data)",
        "    bm.edges.ensure_lookup_table()",
        "    bm.verts.ensure_lookup_table()",
        "",
        "    for edge in bm.edges:",
        "        if not edge.is_manifold or edge.is_boundary:",
        "            edge.smooth = False",
        "            edge.seam = True",
        "            crease_layer = bm.edges.layers.crease.verify()",
        "            edge[crease_layer] = 1.0",
        "",
        "    bm.to_mesh(obj.data)",
        "    bm.free()",
        "    print('CAD-Anchored Mesh Imported and Marked (Sharp Seams Only).')",
    ]
    path.write_text("\n".join(script), encoding="utf-8")

def _fmt_float_for_id(x: float) -> str:
    s = f"{float(x):.6g}"
    return s.replace("-", "m").replace(".", "p")

def _default_run_id(args: argparse.Namespace) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_defl{_fmt_float_for_id(args.deflection)}_p5{args.phase5_mode}"

def _parse_phase_spec(spec: str) -> List[int]:
    """Parse phase spec like '1-5' or '1,3,5'."""
    spec = spec.strip()
    if "-" in spec:
        a, b = spec.split("-", 1)
        lo, hi = int(a), int(b)
        return list(range(min(lo, hi), max(lo, hi) + 1))
    if "," in spec:
        return [int(x.strip()) for x in spec.split(",") if x.strip()]
    return [int(spec)]

def cmd_run(args: argparse.Namespace) -> int:
    # Lazy load CAD dependencies so --help executes cleanly
    from .metrics import flatten_runs_for_csv, format_metrics_table
    from .pipeline import run_pipeline_headless
    from .shading import split_vertices_by_crease
    from .types import RemeshSettings

    step_path = Path(args.step).resolve()
    if not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    run_id = args.run_id.strip() if str(args.run_id).strip() else _default_run_id(args)
    logs_dir = Path(args.logs_dir) / run_id
    demo_out_dir = Path(args.demo_output_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    demo_out_dir.mkdir(parents=True, exist_ok=True)

    out_json = Path(args.json) if args.json else (logs_dir / "metrics.json")
    out_csv = Path(args.csv) if args.csv else (logs_dir / "metrics.csv")
    out_obj_base = Path(args.out_obj) if args.out_obj else (demo_out_dir / f"{run_id}_phase5.obj")
    latest_obj = demo_out_dir / "latest_phase5.obj"

    run_args_path = logs_dir / "run_args.json"
    run_args_payload = {
        "run_id": run_id,
        "step": str(step_path),
        "phases": args.phases,
        "repeat": int(args.repeat),
        "deflection": float(args.deflection),
        "ang_deflection": float(args.ang_deflection),
        "phase5_mode": args.phase5_mode,
        "phase5_keep_largest_component": bool(args.phase5_keep_largest_component),
        "phase5_topo_target_factor": float(args.phase5_topo_target_factor),
        "phase5_topo_remesh_iters": int(args.phase5_topo_remesh_iters),
        "shade_sharp_deg": float(args.shade_sharp_deg),
        "phase5_smooth_weld_enable": bool(args.phase5_smooth_weld_enable),
        "phase5_smooth_weld_tol": float(args.phase5_smooth_weld_tol),
        "phase5_smooth_weld_angle_deg": float(args.phase5_smooth_weld_angle_deg),
        "phase5_normal_relax_iters": int(args.phase5_normal_relax_iters),
        "phase5_normal_relax_strength": float(args.phase5_normal_relax_strength),
        "phase5_normal_relax_global": bool(args.phase5_normal_relax_global),
        "phase5_smooth_weld_global": bool(args.phase5_smooth_weld_global),
        "flip_normals": bool(args.flip_normals),
        "logs_dir": str(logs_dir),
        "demo_output_dir": str(demo_out_dir),
        "json": str(out_json),
        "csv": str(out_csv),
        "out_obj": str(out_obj_base),
    }
    run_args_path.write_text(json.dumps(run_args_payload, indent=2), encoding="utf-8")

    phases = _parse_phase_spec(args.phases)
    all_runs = []

    for r in range(args.repeat):
        out = run_pipeline_headless(
            step_path=step_path,
            enabled_phases=phases,
            tess_linear_deflection=args.deflection,
            tess_angular_deflection=args.ang_deflection,
            phase5_mode=args.phase5_mode,
            phase5_keep_largest_component=args.phase5_keep_largest_component,
            phase5_topo_target_factor=args.phase5_topo_target_factor,
            phase5_topo_remesh_iters=args.phase5_topo_remesh_iters,
            phase5_shade_sharp_deg=args.shade_sharp_deg,
            phase5_flip_normals=args.flip_normals,
            phase5_smooth_weld_enable=args.phase5_smooth_weld_enable,
            phase5_smooth_weld_tol=args.phase5_smooth_weld_tol,
            phase5_smooth_weld_angle_deg=args.phase5_smooth_weld_angle_deg,
            phase5_normal_relax_iters=args.phase5_normal_relax_iters,
            phase5_normal_relax_strength=args.phase5_normal_relax_strength,
            phase5_normal_relax_global=args.phase5_normal_relax_global,
            phase5_smooth_weld_global=args.phase5_smooth_weld_global,
            phase5_edge_snap=args.phase5_edge_snap,
            phase5_face_projection=args.phase5_face_projection,
            remesh_settings=RemeshSettings(
                target_edge_length=args.remesh_resolution,
                max_iters=args.remesh_iters,
                shade_sharp_deg=args.shade_sharp_deg,
                smooth_weld_enable=args.phase5_smooth_weld_enable,
                smooth_weld_tol=args.phase5_smooth_weld_tol,
                smooth_weld_angle_deg=args.phase5_smooth_weld_angle_deg,
                normal_relax_iters=args.phase5_normal_relax_iters,
                normal_relax_strength=args.phase5_normal_relax_strength,
            ) if args.remesh_resolution > 0 else None,
        )
        m = out["metrics"]
        all_runs.append(m)
        print(f"\n=== Run {r + 1}/{args.repeat} ===")
        print(format_metrics_table(m))

        if out["state"].phase5 is not None:
            obj_path = out_obj_base
            if args.repeat > 1:
                obj_path = obj_path.with_name(f"{obj_path.stem}_run{r+1}{obj_path.suffix}")

            p5 = out["state"].phase5
            v_out, f_out, vn_out = p5.V_quad, p5.Q, p5.projected_normals
            if args.shade_sharp_deg > 0.0:
                v_out, f_out, vn_out = split_vertices_by_crease(
                    v_out,
                    f_out,
                    crease_angle_deg=args.shade_sharp_deg,
                    base_normals=vn_out,
                )

            _write_obj(
                obj_path,
                v_out,
                f_out,
                vn_out,
            )
            
            blender_script_path = obj_path.with_name(f"{obj_path.stem}_blender_setup.py")
            _write_blender_script(blender_script_path, obj_path.name, v_out, f_out)
            
            print(f"Saved projected mesh OBJ to: {obj_path}")
            print(f"Saved Blender import script to: {blender_script_path}")
            try:
                shutil.copyfile(obj_path, latest_obj)
                print(f"Updated latest mesh link: {latest_obj}")
            except Exception as exc:
                print(f"[WARN] Failed to update latest mesh alias: {exc}")

    payload = {
        "step_path": str(step_path),
        "repeat": args.repeat,
        "phases": phases,
        "runs": all_runs,
    }

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved JSON metrics to: {out_json}")

    rows = flatten_runs_for_csv(all_runs)
    if rows:
        fields = sorted({k for row in rows for k in row.keys()})
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV metrics to: {out_csv}")
    print(f"Saved run args to: {run_args_path}")

    return 0

def cmd_native_optimize(args: argparse.Namespace) -> int:
    # Lazy load CAD dependencies so --help executes cleanly
    from .metrics import format_metrics_table
    from .native_pipeline import run_native_optimization_pipeline
    from .types import UVAtlasSettings

    step_path = Path(args.step).resolve()
    if not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    run_id = args.run_id.strip() if str(args.run_id).strip() else f"native_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    demo_out_dir = Path(getattr(args, "output_dir", None) or getattr(args, "demo_output_dir", "Demo/output"))
    demo_out_dir.mkdir(parents=True, exist_ok=True)

    out_obj = demo_out_dir / f"{run_id}_native_opt.obj"
    latest_obj = demo_out_dir / "latest_native_opt.obj"

    print(f"\n[NATIVE OPTIMIZE] Starting for: {step_path.name}")
    print(f"  Quality Preset: {args.preset}")
    print(f"  Area Threshold: {args.area_tol}")
    print(f"  Radius Threshold: {args.radius_tol}")
    print(f"  UV Mode: {args.uv_mode}")

    uv_mode = getattr(args, "uv_mode", "SingleTile")
    udim_buckets_raw = getattr(args, "udim_buckets", "1000.0,100.0,10.0")
    try:
        bucket_thresholds = [float(x.strip()) for x in udim_buckets_raw.split(",") if x.strip()]
        if not bucket_thresholds:
            bucket_thresholds = [1000.0, 100.0, 10.0]
    except Exception:
        bucket_thresholds = [1000.0, 100.0, 10.0]

    uv_settings = UVAtlasSettings(
        mode=uv_mode,
        bucket_thresholds_mm2=bucket_thresholds
    ) if (args.unwrap or uv_mode != "SingleTile") else None

    out = run_native_optimization_pipeline(
        step_path=step_path,
        area_threshold=args.area_tol,
        radius_threshold=args.radius_tol,
        linear_deflection=args.deflection,
        angular_deflection=args.ang_deflection,
        preset_name=args.preset,
        validate=args.validate,
        unwrap=args.unwrap or (uv_mode != "SingleTile"),
        uv_settings=uv_settings
    )

    state = out["state"]
    metrics = out["metrics"]

    print("\n=== Native Optimization Metrics ===")
    print(format_metrics_table(metrics))

    if state.phase1 is not None:
        p1 = state.phase1
        _write_obj(out_obj, p1.V, p1.F, VN=p1.VN, VT=p1.UV)
        print(f"Saved optimized OBJ to: {out_obj}")
        try:
            shutil.copyfile(out_obj, latest_obj)
            print(f"Updated latest link: {latest_obj}")
        except Exception as exc:
            print(f"[WARN] Failed to update latest link: {exc}")

    return 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CADOPT: Native CAD-to-Rendering Optimization Engine & CLI Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Command: run (Legacy 5-phase pipeline)
    run = sub.add_parser("run", help="Run 5-phase pipeline headlessly and collect metrics")
    run.add_argument("--step", type=str, default="Demo/sample_asm.stp", help="Path to STEP file (default: Demo/sample_asm.stp)")
    run.add_argument("--phases", type=str, default="1-5", help="Phase selection (e.g., 1-5, 1,3,5)")
    run.add_argument("--repeat", type=int, default=1, help="Repeat count for benchmarking (default: 1)")
    run.add_argument("--deflection", type=float, default=0.15, help="Linear deflection for OCC tessellation (default: 0.15)")
    run.add_argument("--ang-deflection", type=float, default=0.15, help="Angular deflection for OCC tessellation (default: 0.15)")
    run.add_argument("--phase5-mode", type=str, default="tri", choices=["tri", "quad"], help="Phase 5 output mode (default: tri)")
    run.add_argument("--phase5-keep-largest-component", action="store_true", default=True, help="Keep only largest connected component (default: enabled)")
    run.add_argument("--phase5-keep-all-components", dest="phase5_keep_largest_component", action="store_false", help="Keep all connected components")
    run.add_argument("--phase5-topo-target-factor", type=float, default=1.0, help="Topology remesh target edge factor (default: 1.0)")
    run.add_argument("--phase5-topo-remesh-iters", type=int, default=2, help="Iterations for Phase 5 topology uniformization (default: 2)")
    run.add_argument("--remesh-resolution", type=float, default=0.0, help="Phase 2.5 uniform remesh resolution (default: 0.0)")
    run.add_argument("--remesh-iters", type=int, default=5, help="Phase 2.5 uniform remesh iterations (default: 5)")
    run.add_argument("--shade-sharp-deg", type=float, default=45.0, help="Sharp-edge shading split angle in degrees (default: 45.0)")
    run.add_argument("--flip-normals", action="store_true", default=False, help="Manually flip all output normals in Phase 5")
    run.add_argument("--phase5-smooth-weld", dest="phase5_smooth_weld_enable", action="store_true", default=True, help="Enable smooth seam welding (default: on)")
    run.add_argument("--phase5-no-smooth-weld", dest="phase5_smooth_weld_enable", action="store_false", help="Disable smooth seam welding")
    run.add_argument("--phase5-smooth-weld-tol", type=float, default=1e-6, help="Spatial tolerance for smooth seam welding (default: 1e-6)")
    run.add_argument("--phase5-smooth-weld-angle-deg", type=float, default=35.0, help="Max normal angle for smooth weld (default: 35.0)")
    run.add_argument("--phase5-normal-relax-iters", type=int, default=0, help="Normal relaxation iterations (default: 0)")
    run.add_argument("--phase5-normal-relax-strength", type=float, default=0.2, help="Normal relaxation blend strength (default: 0.2)")
    run.add_argument("--phase5-normal-smooth-global", dest="phase5_normal_relax_global", action="store_true", default=False, help="Disable edge-blocking for normal relaxation")
    run.add_argument("--phase5-weld-global", dest="phase5_smooth_weld_global", action="store_true", default=False, help="Merge all coincident vertices regardless of normal similarity")
    run.add_argument("--phase5-edge-snap", action="store_true", default=True, help="Enable boundary-aware edge snapping (default: enabled)")
    run.add_argument("--no-phase5-edge-snap", dest="phase5_edge_snap", action="store_false", help="Disable boundary-aware edge snapping")
    run.add_argument("--phase5-face-projection", action="store_true", default=True, help="Enable analytical face projection (default: enabled)")
    run.add_argument("--no-phase5-face-projection", dest="phase5_face_projection", action="store_false", help="Disable analytical face projection")
    run.add_argument("--run-id", type=str, default="", help="Identifier for organized output folders")
    run.add_argument("--logs-dir", type=str, default="logs", help="Base directory for metrics/log artifacts (default: logs)")
    run.add_argument("--demo-output-dir", type=str, default="Demo/output", help="Directory for exported final output mesh OBJ (default: Demo/output)")
    run.add_argument("--out-obj", type=str, default="", help="Optional OBJ path (default: Demo/output/<run_id>_phase5.obj)")
    run.add_argument("--json", type=str, default="", help="Optional output JSON path")
    run.add_argument("--csv", type=str, default="", help="Optional output CSV path")
    run.set_defaults(func=cmd_run)

    # Command: native-optimize (Canonical NURBS Optimization Engine)
    native_epilog = """
Examples:
  1. Standard Native Optimization with Balanced Preset & UV Unwrap:
     python -m cadopt.cli native-optimize --step Demo/faulhabers_dc_motor.step --preset Balanced --unwrap

  2. Multi-Tile UDIM UV Atlas Packing (Production Bucketed Mode):
     python -m cadopt.cli native-optimize --step Demo/faulhabers_dc_motor.step --preset Balanced --unwrap --uv-mode UDIM-Bucketed --udim-buckets 500.0,50.0,5.0

  3. High-Precision Tessellation with QA Normal Validation:
     python -m cadopt.cli native-optimize --step Demo/mechanical_assembly.stp --preset High --unwrap --validate
"""
    native = sub.add_parser(
        "native-optimize", 
        help="Run streamlined NURBS-native optimization pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=native_epilog
    )
    native.add_argument("--step", type=str, default="Demo/sample_asm.stp", help="Path to STEP input file (default: Demo/sample_asm.stp)")
    native.add_argument("--area-tol", type=float, default=0.0, help="Area threshold in mm^2 for tiny face removal (default: 0.0)")
    native.add_argument("--radius-tol", type=float, default=0.0, help="Radius threshold in mm for micro-fillet removal (default: 0.0)")
    native.add_argument("--deflection", type=float, default=None, help="Linear deflection for tessellation (default: preset-dependent)")
    native.add_argument("--ang-deflection", type=float, default=None, help="Angular deflection in radians for tessellation (default: preset-dependent)")
    native.add_argument("--preset", type=str, default="Balanced", choices=["Draft", "Balanced", "High", "Ultra-CAD", "Anisotropic"], help="Quality preset profile (default: Balanced)")
    native.add_argument("--validate", action="store_true", help="Run QA normal validation using GeometryValidator")
    native.add_argument("--unwrap", action="store_true", help="Generate automated UV atlas coordinates using xatlas")
    native.add_argument("--uv-mode", type=str, default="SingleTile", choices=["SingleTile", "UDIM-Auto", "UDIM-Bucketed"], help="UV Atlas Packing Mode: SingleTile (0-1 canvas), UDIM-Auto (dynamic area ranking), UDIM-Bucketed (surface area threshold cutoffs). (default: SingleTile)")
    native.add_argument("--udim-buckets", type=str, default="1000.0,100.0,10.0", help="Comma-separated surface area cutoff thresholds in mm^2 for UDIM-Bucketed mode (default: 1000.0,100.0,10.0)")
    native.add_argument("--run-id", type=str, default="", help="Optional run identifier string")
    native.add_argument("--output-dir", "--demo-output_dir", dest="demo_output_dir", type=str, default="Demo/output", help="Directory for exported output OBJ mesh (default: Demo/output)")
    native.set_defaults(func=cmd_native_optimize)

    return parser

def main() -> int:
    import traceback
    try:
        parser = build_parser()
        args = parser.parse_args()
        return int(args.func(args))
    except Exception:
        with open("critical_error.log", "w") as f:
            traceback.print_exc(file=f)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())

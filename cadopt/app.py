import time
from pathlib import Path

import polyscope as ps
from polyscope import imgui as psim

from .features import extract_feature_lines
from .pipeline import _transfer_face_ids, solve_orientation_field
from .native_pipeline import run_native_optimization_pipeline, NATIVE_PRESETS
from .field_position import detect_singularities
from .io_step import load_demo_shape
from .quad_extract_project import _remesh_uniform_triangles, extract_and_project_quad_mesh
from .tessellation import tessellate_shape_with_face_mapping
from .types import AppState, RemeshSettings
from .viewer import (
    init_viewer,
    show_phase1_mesh,
    show_phase2_features,
    show_phase3_orientation,
    show_phase4_singularities,
    show_phase5_quad_mesh,
)


class CADAnchoredFieldRemesherApp:
    """Interactive 5-phase prototype orchestrator.

    This class owns the phase state and wires each phase to one GUI button.
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.state = AppState()
        self.remesh_settings = RemeshSettings()
        self.native_preset = "Balanced"
        self.qa_validate = False
        self._layout_seeded = False

    @staticmethod
    def _set_next_window_layout(x: float, y: float, w: float, h: float) -> None:
        """Best-effort fixed window placement to avoid overlapping UI panels."""
        set_pos = getattr(psim, "SetNextWindowPos", None)
        set_size = getattr(psim, "SetNextWindowSize", None)

        if callable(set_pos):
            try:
                set_pos((x, y))
            except TypeError:
                try:
                    set_pos(x, y)
                except Exception:
                    pass

        if callable(set_size):
            try:
                set_size((w, h))
            except TypeError:
                try:
                    set_size(w, h)
                except Exception:
                    pass

    def _safe_begin(self, title: str) -> bool:
        """Handle imgui Begin API variants across polyscope versions."""
        out = psim.Begin(title, True)
        if isinstance(out, tuple):
            expanded, _is_open = out
            return bool(expanded)
        return bool(out)

    def _phase_ready(self, phase_idx: int) -> bool:
        if phase_idx == 1:
            return True
        if phase_idx == 2:
            return self.state.phase1 is not None
        if phase_idx == 3:
            return self.state.phase1 is not None and self.state.phase2 is not None
        if phase_idx == 4:
            return self.state.phase1 is not None and self.state.phase3 is not None
        if phase_idx == 5:
            return self.state.phase1 is not None
        return False

    def run(self) -> None:
        init_viewer()
        # Avoid crowding/conflict with Polyscope default "Structures" + options panels.
        set_default_panels = getattr(ps, "set_build_default_gui_panels", None)
        if callable(set_default_panels):
            try:
                set_default_panels(False)
            except Exception:
                pass
        ps.set_user_callback(self._ui_callback)
        ps.show()

    def _run_phase1(self) -> None:
        step_path, shape = load_demo_shape(self.project_root)
        self.state.step_path = str(step_path)
        self.state.log(f"[Phase 1] Loaded STEP: {step_path}")

        phase1 = tessellate_shape_with_face_mapping(shape)
        self.state.phase1 = phase1
        show_phase1_mesh(self.state, phase1)
        self.state.log(
            f"[Phase 1] Tessellated proxy mesh: |V|={phase1.V.shape[0]}, |F|={phase1.F.shape[0]}, faces={len(phase1.occ_faces)}"
        )

    def _run_phase2(self) -> None:
        if self.state.phase1 is None:
            self.state.log("[Phase 2] Run Phase 1 first.")
            return
        self.state.log("[Phase 2] Extracting G0/G1 features and analytical NURBS flow...")
        phase2 = extract_feature_lines(self.state.phase1)
        self.state.phase2 = phase2
        show_phase2_features(self.state, phase2)
        flows = sum(1 for p in phase2.polylines if p.edge_kind == "flow")
        self.state.log(
            f"[Phase 2] SUCCESS: {len(phase2.polylines)} polylines generated. Analytical fibers: {flows}"
        )

    def _run_phase3(self) -> None:
        if self.state.phase1 is None or self.state.phase2 is None:
            self.state.log("[Phase 3] Run Phase 1 and Phase 2 first.")
            return

        # Phase 2.5: Interactive Regularization
        t0_reg = time.perf_counter()
        v_pre, f_pre = self.state.phase1.V.shape[0], self.state.phase1.F.shape[0]
        
        old_V, old_F = self.state.phase1.V.copy(), self.state.phase1.F.copy()
        old_face_ids = self.state.phase1.tri_to_face_idx.copy()

        Vr, Fr = _remesh_uniform_triangles(
            self.state.phase1.V,
            self.state.phase1.F,
            self.remesh_settings.target_edge_length,
            self.remesh_settings.max_iters,
        )
        self.state.phase1.V = Vr
        self.state.phase1.F = Fr
        
        # Recover face mapping
        self.state.phase1.tri_to_face_idx = _transfer_face_ids(
            old_V, old_F, old_face_ids, Vr, Fr
        )
        
        dt_reg = time.perf_counter() - t0_reg
        self.state.log(f"[Phase 2.5] Proxy regularized: {v_pre}->{Vr.shape[0]} verts in {dt_reg:.3f}s")

        phase3 = solve_orientation_field(self.state.phase1, self.state.phase2)
        self.state.phase3 = phase3
        show_phase3_orientation(self.state, phase3)
        self.state.log("[Phase 3] Orientation field solved and visualized.")

    def _run_phase4(self) -> None:
        if self.state.phase1 is None or self.state.phase3 is None:
            self.state.log("[Phase 4] Run Phase 1 and Phase 3 first.")
            return
        phase4 = detect_singularities(self.state.phase1, self.state.phase3)
        self.state.phase4 = phase4
        show_phase4_singularities(self.state, phase4)
        self.state.log(f"[Phase 4] Singularities found: {phase4.singular_points.shape[0]}")

    def _run_phase5(self) -> None:
        if self.state.phase1 is None:
            self.state.log("[Phase 5] Run Phase 1 first.")
            return

        flines = self.state.phase2.polylines if self.state.phase2 else None
        phase5 = extract_and_project_quad_mesh(
            self.state.phase1,
            feature_lines=flines,
            split_by_face=True,
            shade_sharp_deg=self.remesh_settings.shade_sharp_deg,
            keep_largest_component=True,
            smooth_weld_enable=self.remesh_settings.smooth_weld_enable,
            smooth_weld_tol=self.remesh_settings.smooth_weld_tol,
            smooth_weld_angle_deg=self.remesh_settings.smooth_weld_angle_deg,
            normal_relax_iters=self.remesh_settings.normal_relax_iters,
            normal_relax_strength=self.remesh_settings.normal_relax_strength,
        )
        self.state.phase5 = phase5
        show_phase5_quad_mesh(self.state, phase5)
        self.state.log(
            f"[Phase 5] Quad-stabilized mesh: |Vq|={phase5.V_quad.shape[0]}, |Q|={phase5.Q.shape[0]}"
        )

    def _run_native_optimize(self) -> None:
        step_path, _shape = load_demo_shape(self.project_root)
        self.state.log(f"[Native] Optimizing: {step_path.name}...")
        
        # We use selected preset for high-fidelity optimization
        out = run_native_optimization_pipeline(
            step_path,
            area_threshold=0.01,
            radius_threshold=0.1,
            preset_name=self.native_preset,
            validate=self.qa_validate
        )
        
        self.state.phase1 = out["state"].phase1
        show_phase1_mesh(self.state, self.state.phase1)
        
        m = out["metrics"]["phases"]
        removed = m.get("defeaturing", {}).get("removed_faces", 0)
        v = m.get("tessellation", {}).get("n_verts", 0)
        
        qa = m.get("qa_validation", {})
        if qa:
            status = "PASS" if qa["ok"] else "FAIL"
            self.state.log(f"[Native] SUCCESS: Removed {removed} features. Mesh: {v} verts. QA: {status}")
        else:
            self.state.log(f"[Native] SUCCESS: Removed {removed} features. Mesh: {v} verts.")

    def _ui_callback(self) -> None:
        # Seed layout only once so users can drag/resize windows afterwards.
        if not self._layout_seeded:
            self._set_next_window_layout(24, 24, 500, 560)
            self._layout_seeded = True

        if not self._safe_begin("CADOPT Controls"):
            psim.End()
            return

        psim.TextUnformatted("CAD-Anchored Field Remesher")
        psim.TextUnformatted("5-Phase pipeline")
        psim.Separator()

        psim.TextUnformatted("Phase readiness")
        for i in range(1, 6):
            status = "READY" if self._phase_ready(i) else "WAIT"
            psim.BulletText(f"P{i}: {status}")
        psim.Separator()

        if psim.Button("1. Load & Tessellate"):
            try:
                self._run_phase1()
            except Exception as exc:
                self.state.log(f"[Phase 1] ERROR: {exc}")

        if psim.Button("2. Extract Feature Lines"):
            try:
                self._run_phase2()
            except Exception as exc:
                self.state.log(f"[Phase 2] ERROR: {exc}")

        if psim.Button("3. Solve Orientation"):
            try:
                self._run_phase3()
            except Exception as exc:
                self.state.log(f"[Phase 3] ERROR: {exc}")

        if psim.Button("4. Find Singularities"):
            try:
                self._run_phase4()
            except Exception as exc:
                self.state.log(f"[Phase 4] ERROR: {exc}")

        if psim.Button("5. Extract & Project Quad Mesh"):
            try:
                self._run_phase5()
            except Exception as exc:
                self.state.log(f"[Phase 5] ERROR: {exc}")

        psim.Separator()
        if psim.Button("NATIVE OPTIMIZE (Direct)"):
            try:
                self._run_native_optimize()
            except Exception as exc:
                self.state.log(f"[Native] ERROR: {exc}")

        psim.Separator()
        psim.TextUnformatted("Native Quality Presets")
        if psim.BeginCombo("Preset", self.native_preset):
            for p_name in NATIVE_PRESETS.keys():
                _, selected = psim.Selectable(p_name, self.native_preset == p_name)
                if selected:
                    self.native_preset = p_name
            psim.EndCombo()
            
        changed_qa, new_qa = psim.Checkbox("Run QA Validation", self.qa_validate)
        if changed_qa:
            self.qa_validate = new_qa
            
        psim.Separator()
        psim.TextUnformatted("Remesh Settings (Phase 2.5)")
        
        changed, new_len = psim.InputFloat("Target Edge Length", self.remesh_settings.target_edge_length)
        if changed:
            self.remesh_settings.target_edge_length = float(max(0.001, new_len))
            
        changed_i, new_iters = psim.InputInt("Remesh Iters", self.remesh_settings.max_iters)
        if changed_i:
            self.remesh_settings.max_iters = int(max(0, min(10, new_iters)))

        psim.Separator()
        if self.state.step_path:
            psim.TextWrapped(f"Loaded STEP: {self.state.step_path}")
        else:
            psim.TextWrapped("Loaded STEP: (none)")

        if psim.Button("Clear Log"):
            self.state.messages.clear()

        psim.Separator()
        psim.TextUnformatted("Recent log (latest 12):")
        for msg in self.state.messages[-12:]:
            psim.BulletText(msg)

        psim.End()

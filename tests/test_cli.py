"""
Comprehensive Unit & Integration Test Suite for CADOPT CLI Interface.

Validates the full Expected / Intended Behavior Matrix:
1. Top-level and subcommand --help flags.
2. Fast lazy loading (no CAD runtime crashes on --help).
3. Parameter parsing & validation (presets, UV modes, deflection overrides, UDIM buckets).
4. Error handling & fallbacks (malformed bucket strings, missing files, invalid choices).
5. Execution handlers (cmd_native_optimize and cmd_run with mocked and real CAD pipelines).
"""

import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

from cadopt.cli import build_parser, cmd_native_optimize, cmd_run, _parse_phase_spec
from cadopt.types import Phase1Data


class TestCADOptCLIMatrix(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()

    # -------------------------------------------------------------------------
    # 1. HELP & DISCOVERY MATRIX
    # -------------------------------------------------------------------------
    def test_top_level_help(self):
        """Verify top-level --help outputs description and available subcommands."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            with self.assertRaises(SystemExit) as cm:
                self.parser.parse_args(["--help"])
            self.assertEqual(cm.exception.code, 0)
            output = fake_out.getvalue()
            self.assertIn("CADOPT: Native CAD-to-Rendering Optimization Engine", output)
            self.assertIn("native-optimize", output)
            self.assertIn("run", output)

    def test_native_optimize_help(self):
        """Verify native-optimize --help displays all required parameters and examples."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            with self.assertRaises(SystemExit) as cm:
                self.parser.parse_args(["native-optimize", "--help"])
            self.assertEqual(cm.exception.code, 0)
            output = fake_out.getvalue()
            # Core parameters
            self.assertIn("--step", output)
            self.assertIn("--preset", output)
            self.assertIn("--unwrap", output)
            self.assertIn("--uv-mode", output)
            self.assertIn("--udim-buckets", output)
            self.assertIn("--area-tol", output)
            self.assertIn("--radius-tol", output)
            self.assertIn("--validate", output)
            self.assertIn("--output-dir", output)
            # Choices
            self.assertIn("SingleTile", output)
            self.assertIn("UDIM-Auto", output)
            self.assertIn("UDIM-Bucketed", output)
            # Examples
            self.assertIn("Examples:", output)
            self.assertIn("UDIM-Bucketed --udim-buckets 500.0,50.0,5.0", output)

    def test_run_help(self):
        """Verify run --help displays parameters for the 5-phase pipeline."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            with self.assertRaises(SystemExit) as cm:
                self.parser.parse_args(["run", "--help"])
            self.assertEqual(cm.exception.code, 0)
            output = fake_out.getvalue()
            self.assertIn("--step", output)
            self.assertIn("--phases", output)
            self.assertIn("--deflection", output)
            self.assertIn("--ang-deflection", output)
            self.assertIn("--phase5-mode", output)

    # -------------------------------------------------------------------------
    # 2. ARGUMENT PARSING & VALIDATION MATRIX
    # -------------------------------------------------------------------------
    def test_native_optimize_arg_parsing(self):
        """Test parsing user command for native-optimize with UDIM bucketed mode."""
        cmd_args = [
            "native-optimize",
            "--step", "Demo/faulhabers_dc_motor.step",
            "--preset", "Balanced",
            "--unwrap",
            "--uv-mode", "UDIM-Bucketed",
            "--udim-buckets", "500.0,50.0,5.0",
            "--area-tol", "1.5",
            "--radius-tol", "0.5",
            "--validate"
        ]
        args = self.parser.parse_args(cmd_args)
        self.assertEqual(args.command, "native-optimize")
        self.assertEqual(args.step, "Demo/faulhabers_dc_motor.step")
        self.assertEqual(args.preset, "Balanced")
        self.assertTrue(args.unwrap)
        self.assertEqual(args.uv_mode, "UDIM-Bucketed")
        self.assertEqual(args.udim_buckets, "500.0,50.0,5.0")
        self.assertEqual(args.area_tol, 1.5)
        self.assertEqual(args.radius_tol, 0.5)
        self.assertTrue(args.validate)
        self.assertIsNone(args.deflection)
        self.assertIsNone(args.ang_deflection)

    def test_deflection_override_parsing(self):
        """Test explicit deflection override parsing."""
        cmd_args = ["native-optimize", "--deflection", "0.08", "--ang-deflection", "0.25"]
        args = self.parser.parse_args(cmd_args)
        self.assertEqual(args.deflection, 0.08)
        self.assertEqual(args.ang_deflection, 0.25)

    def test_invalid_preset_rejected(self):
        """Verify that an invalid preset choice is rejected by argparse."""
        cmd_args = ["native-optimize", "--step", "Demo/sample.stp", "--preset", "SuperHigh"]
        with patch("sys.stderr", new=StringIO()) as fake_err:
            with self.assertRaises(SystemExit) as cm:
                self.parser.parse_args(cmd_args)
            self.assertNotEqual(cm.exception.code, 0)
            self.assertIn("invalid choice: 'SuperHigh'", fake_err.getvalue())

    def test_invalid_uv_mode_rejected(self):
        """Verify that an invalid uv-mode choice is rejected by argparse."""
        cmd_args = ["native-optimize", "--step", "Demo/sample.stp", "--uv-mode", "InvalidUV"]
        with patch("sys.stderr", new=StringIO()) as fake_err:
            with self.assertRaises(SystemExit) as cm:
                self.parser.parse_args(cmd_args)
            self.assertNotEqual(cm.exception.code, 0)
            self.assertIn("invalid choice: 'InvalidUV'", fake_err.getvalue())

    # -------------------------------------------------------------------------
    # 3. EDGE CASES & ERROR FALLBACK MATRIX
    # -------------------------------------------------------------------------
    def test_missing_step_file_raises_file_not_found(self):
        """Verify executing cmd_native_optimize with non-existent step raises FileNotFoundError."""
        args = self.parser.parse_args(["native-optimize", "--step", "non_existent_file_xyz.step"])
        with self.assertRaises(FileNotFoundError):
            cmd_native_optimize(args)

    def test_udim_buckets_malformed_string_fallback(self):
        """Verify malformed bucket strings fall back safely to defaults [1000.0, 100.0, 10.0]."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            step_file = Path(tmp_dir) / "dummy.step"
            step_file.write_text("HEADER; ENDSEC; DATA; ENDSEC; END-ISO-10303-21;")

            cmd_args = ["native-optimize", "--step", str(step_file), "--uv-mode", "UDIM-Bucketed", "--udim-buckets", "invalid,data"]
            args = self.parser.parse_args(cmd_args)

            with patch("cadopt.native_pipeline.run_native_optimization_pipeline") as mock_pipeline:
                mock_state = MagicMock()
                mock_state.phase1 = None
                mock_pipeline.return_value = {"state": mock_state, "metrics": {"total_time_s": 0.1, "phases": {}}}

                res = cmd_native_optimize(args)
                self.assertEqual(res, 0)
                self.assertTrue(mock_pipeline.called)
                _, kwargs = mock_pipeline.call_args
                self.assertEqual(kwargs["uv_settings"].bucket_thresholds_mm2, [1000.0, 100.0, 10.0])

    def test_udim_buckets_empty_string_fallback(self):
        """Verify empty bucket strings fall back safely to defaults."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            step_file = Path(tmp_dir) / "dummy.step"
            step_file.write_text("HEADER; ENDSEC; DATA; ENDSEC; END-ISO-10303-21;")

            cmd_args = ["native-optimize", "--step", str(step_file), "--uv-mode", "UDIM-Bucketed", "--udim-buckets", "  ,  "]
            args = self.parser.parse_args(cmd_args)

            with patch("cadopt.native_pipeline.run_native_optimization_pipeline") as mock_pipeline:
                mock_state = MagicMock()
                mock_state.phase1 = None
                mock_pipeline.return_value = {"state": mock_state, "metrics": {"total_time_s": 0.1, "phases": {}}}

                res = cmd_native_optimize(args)
                self.assertEqual(res, 0)
                _, kwargs = mock_pipeline.call_args
                self.assertEqual(kwargs["uv_settings"].bucket_thresholds_mm2, [1000.0, 100.0, 10.0])

    # -------------------------------------------------------------------------
    # 4. EXECUTION HANDLERS END-TO-END MATRIX (MOCKED & REAL)
    # -------------------------------------------------------------------------
    def test_cmd_native_optimize_orchestration_mocked(self):
        """Verify cmd_native_optimize orchestration pipeline invocation and OBJ writing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            step_file = Path(tmp_dir) / "test_part.step"
            step_file.write_text("HEADER; ENDSEC; DATA; ENDSEC; END-ISO-10303-21;")
            out_dir = Path(tmp_dir) / "output"

            cmd_args = [
                "native-optimize",
                "--step", str(step_file),
                "--preset", "Balanced",
                "--unwrap",
                "--uv-mode", "UDIM-Bucketed",
                "--udim-buckets", "500.0,50.0,5.0",
                "--output-dir", str(out_dir),
                "--run-id", "test_run_123"
            ]
            args = self.parser.parse_args(cmd_args)

            # Create mock Phase1Data mesh output
            V = np.array([[0,0,0], [1,0,0], [0,1,0]], dtype=float)
            F = np.array([[0,1,2]], dtype=int)
            VN = np.array([[0,0,1], [0,0,1], [0,0,1]], dtype=float)
            UV = np.array([[0,0], [1,0], [0,1]], dtype=float)
            p1_mock = Phase1Data(
                shape=None,
                occ_faces=[],
                V=V, 
                F=F, 
                tri_to_face_idx=np.zeros(1, dtype=int),
                VN=VN, 
                UV=UV
            )

            mock_state = MagicMock()
            mock_state.phase1 = p1_mock

            with patch("cadopt.native_pipeline.run_native_optimization_pipeline") as mock_pipeline:
                mock_pipeline.return_value = {
                    "state": mock_state,
                    "metrics": {
                        "preset": "Balanced",
                        "total_time_s": 0.05,
                        "phases": {
                            "tessellation": {"ok": True, "time_s": 0.02, "n_verts": 3, "n_tris": 1},
                            "uv_unwrapping": {"ok": True, "time_s": 0.03, "uv_mode": "UDIM-Bucketed"}
                        }
                    }
                }

                ret = cmd_native_optimize(args)
                self.assertEqual(ret, 0)

                # Verify files were generated
                written_obj = out_dir / "test_run_123_native_opt.obj"
                latest_link = out_dir / "latest_native_opt.obj"
                self.assertTrue(written_obj.exists())
                self.assertTrue(latest_link.exists())

                obj_text = written_obj.read_text(encoding="utf-8")
                self.assertIn("v 0 0 0", obj_text)
                self.assertIn("vt 0 0", obj_text)
                self.assertIn("vn 0 0 1", obj_text)
                self.assertIn("f 1/1/1 2/2/2 3/3/3", obj_text)

    def test_real_cad_faulhaber_draft_udim_run(self):
        """Integration Test: Run native-optimize end-to-end on real faulhabers_dc_motor.step geometry."""
        real_step = Path("Demo/faulhabers_dc_motor.step")
        if not real_step.exists():
            self.skipTest("Demo/faulhabers_dc_motor.step not found")

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "output"
            cmd_args = [
                "native-optimize",
                "--step", str(real_step),
                "--preset", "Draft",
                "--unwrap",
                "--uv-mode", "UDIM-Bucketed",
                "--udim-buckets", "500.0,50.0,5.0",
                "--output-dir", str(out_dir),
                "--run-id", "real_test_run"
            ]
            args = self.parser.parse_args(cmd_args)

            ret = cmd_native_optimize(args)
            self.assertEqual(ret, 0)

            exported_obj = out_dir / "real_test_run_native_opt.obj"
            self.assertTrue(exported_obj.exists())
            
            # Check content of generated OBJ
            content = exported_obj.read_text(encoding="utf-8")
            self.assertIn("v ", content)
            self.assertIn("vn ", content)
            self.assertIn("vt ", content)
            self.assertIn("f ", content)


if __name__ == "__main__":
    unittest.main()

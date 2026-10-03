# CADOPT Test Suite & Intended Behavior Test Matrix

This directory contains the automated test suite for the **CADOPT** Native CAD-to-Rendering Optimization Engine and CLI toolkit.

---

## 1. Running the Test Suite

### Run All Unit & Integration Tests via `unittest`
From the project root directory:
```bash
python -m unittest discover -s tests
```

Or run the CLI test matrix directly:
```bash
python -m unittest tests/test_cli.py
```

### Run Tests via `pytest`
```bash
pytest tests/
```

---

## 2. Expected / Intended Behavior Test Matrix

The test matrix in [`test_cli.py`](test_cli.py) validates all CLI parameter combinations, error fallbacks, lazy loading dependencies, and pipeline orchestration.

| Category | Component / Feature | Test Case Scenario | Input / Command | Expected / Intended Behavior | Verification Status |
|---|---|---|---|---|---|
| **Help & Discovery** | Top-level CLI | Help flag execution | `python -m cadopt.cli --help` | Exit code `0`, outputs header, subcommands (`native-optimize`, `run`) | ✅ **PASS** |
| | Subcommand `native-optimize` | Help flag execution | `python -m cadopt.cli native-optimize --help` | Exit code `0`, displays all arguments (`--step`, `--preset`, `--unwrap`, `--uv-mode`, `--udim-buckets`, `--area-tol`, `--radius-tol`, `--validate`, `--output-dir`), choices, and usage examples | ✅ **PASS** |
| | Subcommand `run` | Help flag execution | `python -m cadopt.cli run --help` | Exit code `0`, displays 5-phase pipeline arguments | ✅ **PASS** |
| | Lazy Loading | CAD-less environment safety | `python -m cadopt.cli --help` | Executes instantly (<0.05s) without raising `ModuleNotFoundError` for missing C++ DLLs | ✅ **PASS** |
| **Argument Validation** | Choice Constraints | Invalid quality preset | `--preset SuperHigh` | Exit code $\ne 0$, argparse error: `invalid choice: 'SuperHigh'` | ✅ **PASS** |
| | Choice Constraints | Invalid UV packing mode | `--uv-mode InvalidUV` | Exit code $\ne 0$, argparse error: `invalid choice: 'InvalidUV'` | ✅ **PASS** |
| | Deflection Overrides | Custom tessellation resolution | `--deflection 0.08 --ang-deflection 0.25` | Overrides preset defaults cleanly (`linear_deflection=0.08`, `angular_deflection=0.25`) | ✅ **PASS** |
| **Error Fallbacks & Edge Cases** | Missing STEP File | Non-existent STEP path | `native-optimize --step non_existent_file.step` | Raises explicit `FileNotFoundError` prior to pipeline launch | ✅ **PASS** |
| | UDIM Bucket Formatting | Malformed bucket string | `--udim-buckets "invalid,data"` | Safely catches exception and falls back to default thresholds `[1000.0, 100.0, 10.0]` $\text{mm}^2$ | ✅ **PASS** |
| | UDIM Bucket Formatting | Empty string / whitespace | `--udim-buckets " , "` | Safely falls back to default thresholds `[1000.0, 100.0, 10.0]` $\text{mm}^2$ | ✅ **PASS** |
| **Execution & Orchestration** | Pipeline Integration (Mocked) | OBJ output generation | `cmd_native_optimize` with mock B-Rep mesh | Creates output directory, exports `.obj` with `v`, `vt`, `vn`, and `f` indexes, creates `latest_native_opt.obj` link | ✅ **PASS** |
| | Pipeline Integration (Real CAD) | Real STEP file processing | `native-optimize --step Demo/faulhabers_dc_motor.step --preset Draft --unwrap --uv-mode UDIM-Bucketed --udim-buckets 500.0,50.0,5.0` | Executes full pipeline on real motor geometry: B-Rep healing $\rightarrow$ tessellation $\rightarrow$ 4-tile UDIM unwrapping $\rightarrow$ `.obj` export | ✅ **PASS** |

---

## 3. Test Coverage Summary

- **CLI Discovery**: Verified that `--help` works cleanly across top-level and subcommands without requiring heavy OpenCASCADE C++ runtime bindings at import time.
- **Deflection & Preset Overrides**: Verified that preset quality profiles (`Draft`, `Balanced`, `High`, `Ultra-CAD`, `Anisotropic`) map to chordal/angular tolerances, and explicit user `--deflection` overrides are respected.
- **UDIM Multi-Tile Packing**: Verified that `--uv-mode UDIM-Bucketed` and `--udim-buckets` parse thresholds correctly and allocate CAD patches into UDIM tile buckets (`1001`, `1002`, `1003`, etc.).
- **OBJ Format Integrity**: Verified that exported Wavefront `.obj` files contain synchronized positions (`v`), analytical normals (`vn`), texture coordinates (`vt`), and face indexes (`f v/vt/vn`).

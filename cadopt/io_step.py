from __future__ import annotations

from pathlib import Path
from typing import Tuple

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone


def find_demo_step_file(demo_dir: Path) -> Path:
    """Find the first STEP file in ./Demo.

    The project directive requires hardcoded ingestion from ./Demo.
    """
    if not demo_dir.exists():
        raise FileNotFoundError(
            f"Demo directory not found: {demo_dir}. Create it and place a .stp/.step file inside."
        )

    step_candidates = sorted(
        [p for p in demo_dir.iterdir() if p.suffix.lower() in {".stp", ".step"}]
    )
    if not step_candidates:
        raise FileNotFoundError(
            f"No STEP file found in {demo_dir}. Expected one .stp or .step file."
        )
    return step_candidates[0]


def load_step_shape(step_path: Path):
    """Load STEP into a TopoDS_Shape via OpenCascade STEP reader."""
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed reading STEP file: {step_path}")

    transfer_ok = reader.TransferRoots()
    if transfer_ok == 0:
        raise RuntimeError(f"STEP transfer failed: {step_path}")

    shape = reader.OneShape()
    return shape


def load_demo_shape(project_root: Path) -> Tuple[Path, object]:
    """Convenience wrapper: find ./Demo/*.stp and load it."""
    demo_dir = project_root / "Demo"
    step_path = find_demo_step_file(demo_dir)
    shape = load_step_shape(step_path)
    return step_path, shape

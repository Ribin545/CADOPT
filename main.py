from __future__ import annotations

from pathlib import Path

from cadopt.app import CADAnchoredFieldRemesherApp


def main() -> None:
    project_root = Path(__file__).resolve().parent
    app = CADAnchoredFieldRemesherApp(project_root=project_root)
    app.run()


if __name__ == "__main__":
    main()

"""Environment diagnostics for PoseSense.

Reports the Python version, installed library versions, model file status,
and basic system information.  Useful for troubleshooting installation issues.

Can be run directly::

    python scripts/check_environment.py

Or via the CLI::

    python -m posesense check
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path


def _check_import(name: str) -> str:
    """Try to import a package and return its version string."""
    try:
        mod = __import__(name)
        version: str = getattr(mod, "__version__", "installed (version unknown)")
        return version
    except ImportError:
        return "NOT INSTALLED"


def run_checks() -> None:
    """Print a comprehensive environment report."""
    print("=" * 60)
    print("PoseSense Environment Check")
    print("=" * 60)

    # ── Python ───────────────────────────────────────────────────────
    print(f"\nPython:   {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine:  {platform.machine()}")
    print(f"CWD:      {Path.cwd()}")

    # ── Required libraries ───────────────────────────────────────────
    print("\n--- Required Libraries ---")
    libs = [
        ("mediapipe", "mediapipe"),
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("joblib", "joblib"),
        ("matplotlib", "matplotlib"),
    ]
    all_ok = True
    for display_name, import_name in libs:
        ver = _check_import(import_name)
        status = "[OK]" if ver != "NOT INSTALLED" else "[FAIL]"
        if ver == "NOT INSTALLED":
            all_ok = False
        print(f"  {status} {display_name:14s} {ver}")

    # ── Dev libraries ────────────────────────────────────────────────
    print("\n--- Dev Libraries ---")
    dev_libs = [
        ("pytest", "pytest"),
        ("pytest-cov", "pytest_cov"),
        ("ruff", "ruff"),
        ("mypy", "mypy"),
    ]
    for display_name, import_name in dev_libs:
        ver = _check_import(import_name)
        status = "[OK]" if ver != "NOT INSTALLED" else "[-]"
        print(f"  {status} {display_name:14s} {ver}")

    # ── Tkinter ──────────────────────────────────────────────────────
    print("\n--- GUI ---")
    try:
        import tkinter

        tk_ver = tkinter.TkVersion
        print(f"  [OK] tkinter        {tk_ver}")
    except ImportError:
        print("  [FAIL] tkinter        NOT AVAILABLE")
        all_ok = False

    # ── MediaPipe model file ─────────────────────────────────────────
    print("\n--- MediaPipe Model ---")
    model_paths = [
        Path("models/mediapipe/pose_landmarker_lite.task"),
        Path("models/mediapipe/pose_landmarker_full.task"),
        Path("models/mediapipe/pose_landmarker_heavy.task"),
    ]
    any_model = False
    for mp in model_paths:
        if mp.exists():
            size_mb = mp.stat().st_size / (1024 * 1024)
            print(f"  [OK] {mp}  ({size_mb:.1f} MiB)")
            any_model = True
        else:
            print(f"  [-] {mp}  (not found)")

    if not any_model:
        print("\n  [WARN] No model file found. Run:")
        print("     python -m posesense download-model --variant lite")

    # ── Data directories ─────────────────────────────────────────────
    print("\n--- Data Directories ---")
    for d in [Path("data"), Path("data/datasets"), Path("data/logs"),
              Path("models"), Path("artifacts")]:
        status = "exists" if d.exists() else "will be created"
        print(f"  {str(d):30s} {status}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_ok:
        print("All required libraries are installed.")
    else:
        print("Some required libraries are missing. Run:")
        print("  uv sync")
        print("or:")
        print("  pip install -e '.[dev]'")
    print("=" * 60)


if __name__ == "__main__":
    run_checks()

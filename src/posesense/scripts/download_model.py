"""Download the MediaPipe PoseLandmarker model file.

Downloads the ``.task`` model bundle into ``models/mediapipe/``.

Verified model URLs (as of the MediaPipe Tasks API documentation)::

    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

Usage::

    python -m posesense download-model --variant lite
    python scripts/download_model.py --variant full --force
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger("posesense.download_model")

_BASE_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_{variant}/float16/1/"
    "pose_landmarker_{variant}.task"
)

_VALID_VARIANTS = ("lite", "full", "heavy")
_DEST_DIR = Path("models") / "mediapipe"


def _get_url(variant: str) -> str:
    """Return the verified download URL for the given variant."""
    if variant not in _VALID_VARIANTS:
        msg = f"Invalid variant '{variant}'; must be one of {_VALID_VARIANTS}"
        raise ValueError(msg)
    return _BASE_URL.format(variant=variant)


def download_model(
    variant: str = "lite",
    dest_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Download the PoseLandmarker model bundle.

    Args:
        variant: Model variant (``lite``, ``full``, ``heavy``).
        dest_dir: Target directory.  Defaults to ``models/mediapipe/``.
        force: Re-download even if the file already exists.

    Returns:
        Path to the downloaded ``.task`` file.

    Raises:
        ValueError: If the variant is invalid.
        RuntimeError: If the download fails.
    """
    if dest_dir is None:
        dest_dir = _DEST_DIR

    url = _get_url(variant)
    filename = f"pose_landmarker_{variant}.task"
    dest_path = dest_dir / filename

    if dest_path.exists() and not force:
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"Model already exists: {dest_path} ({size_mb:.1f} MiB)")
        print("Use --force to re-download.")
        return dest_path

    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {variant} model from:")
    print(f"  {url}")
    print(f"To: {dest_path}")

    # Download to a temporary file first, then rename (atomic write).
    try:
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            dir=str(dest_dir), suffix=".task.tmp"
        )
        tmp_path = Path(tmp_path_str)

        # Use a simple progress indicator
        def _reporthook(
            block_num: int, block_size: int, total_size: int
        ) -> None:
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                print(f"\r  Progress: {pct}% ({mb:.1f} / {total_mb:.1f} MiB)", end="", flush=True)
            else:
                mb = downloaded / (1024 * 1024)
                print(f"\r  Downloaded: {mb:.1f} MiB", end="", flush=True)

        import os

        os.close(tmp_fd)
        urllib.request.urlretrieve(url, str(tmp_path), reporthook=_reporthook)
        print()  # newline after progress

        # Verify non-empty
        if tmp_path.stat().st_size == 0:
            tmp_path.unlink(missing_ok=True)
            msg = "Downloaded file is empty — the URL may be wrong or the network interrupted."
            raise RuntimeError(msg)

        # Atomic rename
        tmp_path.replace(dest_path)

        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"[OK] Downloaded: {dest_path} ({size_mb:.1f} MiB)")
        return dest_path

    except urllib.error.URLError as exc:
        print(f"\n[FAIL] Download failed: {exc}", file=sys.stderr)
        print("\nManual download instructions:", file=sys.stderr)
        print(f"  1. Open in browser: {url}", file=sys.stderr)
        print(f"  2. Save the file to: {dest_path}", file=sys.stderr)
        raise RuntimeError(f"Download failed: {exc}") from exc
    except OSError as exc:
        print(f"\n[FAIL] File system error: {exc}", file=sys.stderr)
        raise RuntimeError(f"File write error: {exc}") from exc
    finally:
        # Clean up partial download
        if "tmp_path" in locals() and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def main() -> None:
    """CLI entry point when run directly."""
    parser = argparse.ArgumentParser(
        description="Download the MediaPipe PoseLandmarker model"
    )
    parser.add_argument(
        "--variant",
        choices=list(_VALID_VARIANTS),
        default="lite",
        help="Model variant (default: lite)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file exists",
    )
    args = parser.parse_args()
    download_model(variant=args.variant, force=args.force)


if __name__ == "__main__":
    main()

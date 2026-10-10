"""Tests for the download_model script."""

from __future__ import annotations

from pathlib import Path

import pytest

from posesense.scripts.download_model import _get_url, download_model


class TestGetUrl:
    """Test URL generation for model variants."""

    def test_lite_url(self) -> None:
        url = _get_url("lite")
        assert "pose_landmarker_lite" in url
        assert url.startswith("https://storage.googleapis.com/")
        assert url.endswith(".task")

    def test_full_url(self) -> None:
        url = _get_url("full")
        assert "pose_landmarker_full" in url

    def test_heavy_url(self) -> None:
        url = _get_url("heavy")
        assert "pose_landmarker_heavy" in url

    def test_invalid_variant(self) -> None:
        with pytest.raises(ValueError, match="Invalid variant"):
            _get_url("mega")


class TestDownloadModel:
    """Test download logic (without actually downloading)."""

    def test_skip_existing(self, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        """If the file exists, skip download."""
        dest_dir = tmp_path / "models" / "mediapipe"
        dest_dir.mkdir(parents=True)
        existing = dest_dir / "pose_landmarker_lite.task"
        existing.write_bytes(b"fake model content")

        result = download_model(variant="lite", dest_dir=dest_dir, force=False)
        assert result == existing
        captured = capsys.readouterr()
        assert "already exists" in captured.out

    def test_creates_directory(self, tmp_path: Path) -> None:
        """The destination directory should be created if it doesn't exist."""
        dest_dir = tmp_path / "new" / "deep" / "path"
        # Don't actually download — just verify directory creation logic
        # by checking that download_model raises a RuntimeError (network unavailable)
        # but the directory was created
        try:
            download_model(variant="lite", dest_dir=dest_dir, force=False)
        except RuntimeError:
            pass  # Expected — no network in tests
        # Even on failure, the directory should have been created
        assert dest_dir.exists()

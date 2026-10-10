"""Tests for the CLI."""

from __future__ import annotations

import pytest

from posesense.cli import main


class TestCLI:
    """Test CLI argument parsing and basic commands."""

    def test_version(self, capsys) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(SystemExit, match="0"):
            main(["--version"])
        captured = capsys.readouterr()
        assert "posesense" in captured.out
        assert "0.1.0" in captured.out

    def test_no_command_shows_help(self, capsys) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(SystemExit, match="0"):
            main([])

    def test_check_command(self, capsys) -> None:  # type: ignore[no-untyped-def]
        """Ensure `check` runs without crashing."""
        main(["check"])
        captured = capsys.readouterr()
        assert "PoseSense Environment Check" in captured.out
        assert "Python:" in captured.out

    def test_placeholder_commands(self, capsys) -> None:  # type: ignore[no-untyped-def]
        """Placeholder commands should exit cleanly."""
        for cmd in ["live", "gui"]:
            with pytest.raises(SystemExit, match="0"):
                main([cmd])

    def test_download_model_invalid_variant(self) -> None:
        with pytest.raises(SystemExit):
            main(["download-model", "--variant", "mega"])

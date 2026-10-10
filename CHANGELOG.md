# Changelog

All notable changes to PoseSense will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Phase 0: Foundation
- Repository scaffold with full directory structure
- `pyproject.toml` with PEP 621 metadata and pinned dependencies
- Configuration system (`config.py`, `config/default.toml`) with typed dataclasses and validation
- Typed exception hierarchy (`exceptions.py`) covering camera, detector, features, data, ML, and config errors
- Shared immutable types (`types.py`): `AppState`, `Landmark`, `PoseResult`, `QualityReport`, `Prediction`, `FrameResult`
- Structured logging setup with rotating file handler
- CLI skeleton with `check` and `download-model` subcommands
- `scripts/check_environment.py` — reports Python/library versions and model file status
- `scripts/download_model.py` — downloads MediaPipe PoseLandmarker model with atomic writes
- `.pre-commit-config.yaml` with ruff, mypy, and pre-commit hooks
- `.gitignore` covering model files, user data, and artifacts
- Initial test suite for config validation, exceptions, types, logging, and CLI
- MIT License, CHANGELOG, README, and documentation stubs

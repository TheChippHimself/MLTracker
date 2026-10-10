"""Shared pytest configuration and fixtures for PoseSense tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the src directory is on the path for imports.
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


@pytest.fixture()
def tmp_config(tmp_path: Path) -> Path:
    """Write a valid default.toml to a temporary directory and return its path."""
    config_text = '''\
[camera]
index = 0
width = 640
height = 480
backend = "auto"
mirror_display = true
reconnect_attempts = 3
reconnect_backoff_s = 1.0
black_frame_threshold = 3.0
black_frame_max_consecutive = 30

[detector]
model_variant = "lite"
model_path = "models/mediapipe/pose_landmarker_lite.task"
running_mode = "video"
num_poses = 1
min_pose_detection_confidence = 0.5
min_pose_presence_confidence = 0.5
min_tracking_confidence = 0.5

[quality]
min_visibility = 0.5
min_presence = 0.5
edge_margin = 0.02
min_torso_length_norm = 0.08
required_landmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

[features]
schema_version = 1
coordinate_source = "image"
use_z = false
center = "mid_hip"
scale = "torso_length"
rotate_to_torso = false
mirror_augmentation = false

[smoothing]
landmark_filter = "none"
ema_alpha = 0.5
vote_window = 7
min_votes = 4
no_person_debounce_frames = 5

[rejection]
min_confidence = 0.60
min_margin = 0.15

[training]
random_seed = 42
default_model = "random_forest"
test_session_fraction = 0.2
min_samples_per_class = 30
imbalance_warning_ratio = 3.0

[performance]
target_fps = 15
log_interval_s = 5

[privacy]
record_video = false
log_predictions = false
'''
    config_path = tmp_path / "config.toml"
    config_path.write_text(config_text, encoding="utf-8")
    return config_path

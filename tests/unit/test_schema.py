"""Tests for the FeatureExtractor schema."""

from __future__ import annotations

import pytest
from tests.fixtures import make_standing_landmarks, make_tpose_landmarks

from posesense.config import FeaturesConfig
from posesense.exceptions import FeatureSchemaError
from posesense.features.schema import SCHEMA_VERSION
from posesense.features.extractor import FeatureExtractor
from posesense.types import PoseResult


class TestFeatureExtractor:
    """Test normalized 1D vector generation."""

    @pytest.fixture()
    def config(self) -> FeaturesConfig:
        return FeaturesConfig(
            schema_version=SCHEMA_VERSION,
            coordinate_source="image",
            use_z=False,
            center="mid_hip",
            scale="torso_length",
        )

    def test_schema_version_mismatch(self) -> None:
        cfg = FeaturesConfig(schema_version=999)
        with pytest.raises(FeatureSchemaError, match="code only supports"):
            FeatureExtractor(cfg)

    def test_extract_produces_consistent_length(self, config: FeaturesConfig) -> None:
        extractor = FeatureExtractor(config)

        lms_stand = make_standing_landmarks()
        res_stand = PoseResult(landmarks=lms_stand, world_landmarks=None, timestamp_ms=0, frame_width=100, frame_height=100)
        feat_stand = extractor.extract(res_stand)

        lms_tpose = make_tpose_landmarks()
        res_tpose = PoseResult(landmarks=lms_tpose, world_landmarks=None, timestamp_ms=0, frame_width=100, frame_height=100)
        feat_tpose = extractor.extract(res_tpose)

        assert len(feat_stand) == len(feat_tpose)
        assert set(feat_stand.keys()) == set(feat_tpose.keys())

    def test_get_feature_names(self, config: FeaturesConfig) -> None:
        extractor = FeatureExtractor(config)
        names = extractor.get_feature_names()

        assert "nose_x" in names
        assert "nose_y" in names
        assert "angle_left_elbow" in names
        assert "wrist_to_wrist" in names

        if config.use_z:
            assert "nose_z" in names
        else:
            assert "nose_z" not in names

    def test_z_coordinate_inclusion(self) -> None:
        cfg = FeaturesConfig(schema_version=SCHEMA_VERSION, use_z=True)
        extractor = FeatureExtractor(cfg)

        lms = make_standing_landmarks()
        res = PoseResult(landmarks=lms, world_landmarks=None, timestamp_ms=0, frame_width=100, frame_height=100)
        feat = extractor.extract(res)

        assert "nose_z" in feat

    def test_tpose_angles(self, config: FeaturesConfig) -> None:
        extractor = FeatureExtractor(config)
        lms = make_tpose_landmarks()
        res = PoseResult(landmarks=lms, world_landmarks=None, timestamp_ms=0, frame_width=100, frame_height=100)
        feat = extractor.extract(res)

        # In a perfect T-pose, elbow and shoulder angles should be ~180 or ~90 depending on definition
        # Arms are extended straight, so elbow should be ~180
        assert feat["angle_left_elbow"] > 170.0
        assert feat["angle_right_elbow"] > 170.0

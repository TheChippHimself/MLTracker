"""Tests for the Dataset manager."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from posesense.data.dataset import PoseDataset
from posesense.exceptions import LabelError


class TestPoseDataset:
    """Test CSV dataset operations."""

    @pytest.fixture()
    def dataset(self, tmp_path: Path) -> PoseDataset:
        return PoseDataset(data_dir=tmp_path)

    def test_create_session_id(self, dataset: PoseDataset) -> None:
        sid1 = dataset.create_session_id()
        sid2 = dataset.create_session_id()
        assert len(sid1) == 8
        assert sid1 != sid2

    def test_invalid_label_raises_error(self, dataset: PoseDataset) -> None:
        with pytest.raises(LabelError):
            dataset._get_path_for_label("???")

    def test_append_creates_file_with_headers(self, dataset: PoseDataset) -> None:
        features = ["f1", "f2"]
        sample = {"f1": 0.5, "f2": 0.6, "timestamp": 1234}

        dataset.append_samples("test_pose", "sess1", features, [sample])

        csv_path = dataset.raw_dir / "test_pose.csv"
        assert csv_path.exists()

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["timestamp", "session_id", "label", "f1", "f2"]

            row = next(reader)
            assert row == ["1234", "sess1", "test_pose", "0.5", "0.6"]

    def test_append_multiple_samples(self, dataset: PoseDataset) -> None:
        features = ["f1"]
        samples = [
            {"f1": 0.1, "timestamp": 1},
            {"f1": 0.2, "timestamp": 2},
        ]

        dataset.append_samples("pose", "s1", features, samples)

        with open(dataset.raw_dir / "pose.csv", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 3  # header + 2 rows

    def test_load_all_data(self, dataset: PoseDataset) -> None:
        features = ["f1"]
        dataset.append_samples("pose_a", "s1", features, [{"f1": 0.1, "timestamp": 1}])
        dataset.append_samples("pose_b", "s2", features, [{"f1": 0.2, "timestamp": 2}])

        data = dataset.load_all_data()
        assert len(data) == 2

        # Check types
        assert isinstance(data[0]["f1"], float)
        assert data[0]["label"] in ("pose_a", "pose_b")

    def test_get_stats(self, dataset: PoseDataset) -> None:
        features = ["f1"]
        dataset.append_samples("pose_a", "s1", features, [{"f1": 0.1, "timestamp": 1}])
        dataset.append_samples("pose_a", "s1", features, [{"f1": 0.2, "timestamp": 2}])
        dataset.append_samples("pose_a", "s2", features, [{"f1": 0.3, "timestamp": 3}])
        dataset.append_samples("pose_b", "s3", features, [{"f1": 0.4, "timestamp": 4}])

        stats = dataset.get_stats()
        assert stats["pose_a"]["samples"] == 3
        assert stats["pose_a"]["sessions"] == 2
        assert stats["pose_b"]["samples"] == 1
        assert stats["pose_b"]["sessions"] == 1

    def test_export_and_import(self, dataset: PoseDataset, tmp_path: Path) -> None:
        features = ["f1"]
        dataset.append_samples("pose_a", "s1", features, [{"f1": 0.1, "timestamp": 1}])
        
        export_file = tmp_path / "export.csv"
        dataset.export_dataset(export_file)
        assert export_file.exists()

        new_dataset = PoseDataset(data_dir=tmp_path / "new_data")
        new_dataset.import_dataset(export_file)

        stats = new_dataset.get_stats()
        assert "pose_a" in stats
        assert stats["pose_a"]["samples"] == 1

    def test_validate_dataset(self, dataset: PoseDataset) -> None:
        features = ["f1", "f2"]
        dataset.append_samples("pose_a", "s1", features, [{"f1": 0.1, "f2": 0.2, "timestamp": 1}])
        
        errors = dataset.validate_dataset(["f1", "f2"])
        assert len(errors) == 0
        
        errors = dataset.validate_dataset(["f1"])
        assert len(errors) == 1
        assert "feature mismatch" in errors[0]

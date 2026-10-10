"""Tests for model training and evaluation end-to-end."""

import shutil
from pathlib import Path

import pytest

from posesense.config import FeaturesConfig
from posesense.data.dataset import PoseDataset
from posesense.features.extractor import FeatureExtractor
from posesense.model.trainer import Trainer


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> PoseDataset:
    ds = PoseDataset(data_dir=tmp_path)
    cfg = FeaturesConfig()
    extractor = FeatureExtractor(cfg)
    feature_names = extractor.get_feature_names()
    
    # Create synthetic data: 2 classes, 2 sessions per class to allow splitting
    for session_id in ["s1", "s2"]:
        samples = []
        for i in range(15):
            vec = {name: 1.0 for name in feature_names}
            samples.append(vec)
        ds.append_samples("ClassA", session_id, feature_names, samples)

    for session_id in ["s3", "s4"]:
        samples = []
        for i in range(15):
            vec = {name: -1.0 for name in feature_names}
            samples.append(vec)
        ds.append_samples("ClassB", session_id, feature_names, samples)
        
    return ds


def test_trainer_end_to_end(synthetic_dataset: PoseDataset) -> None:
    cfg = FeaturesConfig()
    extractor = FeatureExtractor(cfg)
    trainer = Trainer(synthetic_dataset, extractor)
    
    run_dir = trainer.train(model_type="rf")
    
    assert run_dir is not None
    assert run_dir.exists()
    assert (run_dir / "model.joblib").exists()
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "confusion_matrix.png").exists()

def test_trainer_svm(synthetic_dataset: PoseDataset) -> None:
    cfg = FeaturesConfig()
    extractor = FeatureExtractor(cfg)
    trainer = Trainer(synthetic_dataset, extractor)
    
    run_dir = trainer.train(model_type="svm")
    
    assert run_dir is not None
    assert run_dir.exists()
    assert (run_dir / "model.joblib").exists()

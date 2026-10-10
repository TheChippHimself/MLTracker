"""Tests for model artifact serialization."""

import json
from pathlib import Path

import pytest
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from posesense.exceptions import ModelLoadError
from posesense.features.schema import SCHEMA_VERSION
from posesense.model.artifact import ModelArtifact


def test_artifact_save_load(tmp_path: Path) -> None:
    artifact = ModelArtifact(tmp_path)
    model = Pipeline([("clf", SVC())])
    feature_names = ["a", "b"]
    classes = ["Standing", "Sitting"]
    metrics = {"accuracy": 0.9}
    
    artifact.save(model, feature_names, classes, metrics)
    
    assert (tmp_path / ModelArtifact.MODEL_FILENAME).exists()
    assert (tmp_path / ModelArtifact.METADATA_FILENAME).exists()
    
    loaded_model, metadata = artifact.load()
    assert isinstance(loaded_model, Pipeline)
    assert metadata["feature_names"] == feature_names
    assert metadata["classes"] == classes
    assert metadata["schema_version"] == SCHEMA_VERSION


def test_artifact_schema_mismatch(tmp_path: Path) -> None:
    artifact = ModelArtifact(tmp_path)
    model = Pipeline([("clf", SVC())])
    
    # Save it normally
    artifact.save(model, ["a"], ["A"], {})
    
    # Tamper with metadata to cause mismatch
    meta_path = tmp_path / ModelArtifact.METADATA_FILENAME
    with open(meta_path, "r") as f:
        meta = json.load(f)
    meta["schema_version"] = 999
    with open(meta_path, "w") as f:
        json.dump(meta, f)
        
    with pytest.raises(ModelLoadError, match="incompatible"):
        artifact.load()

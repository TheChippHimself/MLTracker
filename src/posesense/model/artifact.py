"""Model artifact serialization and loading with metadata validation."""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from posesense import __version__
from posesense.exceptions import ModelLoadError
from posesense.features.schema import SCHEMA_VERSION

logger = logging.getLogger(__name__)


class ModelArtifact:
    """Handles saving and loading models with metadata validation."""

    MODEL_FILENAME = "model.joblib"
    METADATA_FILENAME = "metadata.json"

    def __init__(self, run_dir: Path | str) -> None:
        """Initialize the artifact manager with a directory path."""
        self.run_dir = Path(run_dir)

    def save(self, model: Pipeline, feature_names: list[str], classes: list[str], metrics: dict[str, float]) -> None:
        """Save the model and its metadata to the run directory."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = self.run_dir / self.MODEL_FILENAME
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            "posesense_version": __version__,
            "schema_version": SCHEMA_VERSION,
            "feature_names": feature_names,
            "classes": classes,
            "metrics": metrics
        }
        
        meta_path = self.run_dir / self.METADATA_FILENAME
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info("Saved model artifact to %s", self.run_dir)

    def load(self) -> tuple[Pipeline, dict[str, Any]]:
        """Load the model and metadata, validating schema compatibility."""
        model_path = self.run_dir / self.MODEL_FILENAME
        meta_path = self.run_dir / self.METADATA_FILENAME
        
        if not model_path.exists() or not meta_path.exists():
            raise ModelLoadError(str(self.run_dir), "Model files or metadata missing.")
            
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        # Compatibility check
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ModelLoadError(
                str(self.run_dir),
                f"Model schema version ({metadata.get('schema_version')}) "
                f"is incompatible with current code schema ({SCHEMA_VERSION})."
            )
            
        model = joblib.load(model_path)
        return model, metadata

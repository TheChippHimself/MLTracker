"""Pose classification inference with rejection rules."""

from pathlib import Path
from typing import Optional

import pandas as pd
from posesense.exceptions import ModelLoadError
from posesense.features.schema import validate_vector
from posesense.model.artifact import ModelArtifact


class PoseClassifier:
    """Wrapper around a trained scikit-learn model for live inference."""

    def __init__(
        self,
        model_path: Path | str,
        min_confidence: float = 0.5,
        min_margin: float = 0.1,
    ) -> None:
        """Initialize the classifier and load the model artifact."""
        self.model_path = Path(model_path)
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        
        artifact = ModelArtifact(self.model_path)
        try:
            self.model, self.metadata = artifact.load()
        except Exception as e:
            raise ModelLoadError(str(model_path), str(e)) from e
            
        self.feature_names = self.metadata["feature_names"]
        self.classes = self.metadata["classes"]
        
    def predict(self, feature_vector: dict[str, float]) -> tuple[Optional[str], float]:
        """Predict the class of a feature vector with rejection logic.
        
        Returns:
            A tuple of (predicted_label, score). If rejected, label is None.
        """
        validate_vector(feature_vector, self.feature_names)
        
        # Scikit-learn pipelines expect a 2D array or DataFrame
        df = pd.DataFrame([feature_vector], columns=self.feature_names)
        
        # Predict probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(df)[0]
        else:
            # If SVM without probability=True, we can't reliably reject by score
            # We just return the predicted class with score 1.0
            label_idx = self.model.predict(df)[0]
            label = self.classes[label_idx] if isinstance(label_idx, int) else label_idx
            return label, 1.0
            
        # Get top 2 probabilities
        sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        top1_idx, top1_score = sorted_probs[0]
        top2_score = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
        
        top1_label = self.classes[top1_idx]
        
        # Rejection rule
        if top1_score < self.min_confidence or (top1_score - top2_score) < self.min_margin:
            return None, top1_score
            
        return top1_label, top1_score

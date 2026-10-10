"""Trainer for pose classification models."""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from posesense.data.dataset import PoseDataset
from posesense.features.extractor import FeatureExtractor
from posesense.model.artifact import ModelArtifact
from posesense.model.evaluator import Evaluator
from posesense.model.splits import create_train_test_split

logger = logging.getLogger(__name__)


class Trainer:
    """Manages the ML training pipeline."""

    def __init__(self, dataset: PoseDataset, extractor: FeatureExtractor) -> None:
        self.dataset = dataset
        self.extractor = FeatureExtractor(extractor.config)
        self.feature_names = self.extractor.get_feature_names()

    def train(self, model_type: str = "rf") -> Path | None:
        """Train a model, evaluate it, and save the artifact.
        
        Args:
            model_type: 'rf' for Random Forest, 'svm' for Support Vector Machine.
            
        Returns:
            Path to the run directory if successful, None otherwise.
        """
        all_data = self.dataset.load_all_data()
        if not all_data:
            logger.error("Dataset is empty. Cannot train.")
            return None
            
        df = pd.DataFrame(all_data)
        
        # Check if we have enough sessions for splitting
        unique_sessions = df["session_id"].nunique()
        if unique_sessions < 2:
            logger.warning("Only 1 recording session found. Train/test split will leak data or fail.")
            # Fallback to normal split in this emergency case just so it doesn't crash
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
        else:
            train_df, test_df = create_train_test_split(df, test_size=0.2, random_state=42)

        X_train = train_df[self.feature_names]
        y_train = train_df["label"]
        groups_train = train_df["session_id"]
        
        X_test = test_df[self.feature_names]
        y_test = test_df["label"]

        if model_type == "rf":
            base_estimator = RandomForestClassifier(random_state=42, class_weight="balanced")
            param_grid = {
                "classifier__n_estimators": [50, 100],
                "classifier__max_depth": [None, 10, 20]
            }
        elif model_type == "svm":
            base_estimator = SVC(random_state=42, class_weight="balanced", probability=True)
            param_grid = {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__kernel": ["rbf", "linear"]
            }
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("classifier", base_estimator)
        ])

        # GridSearchCV with GroupKFold
        # Ensure we have enough groups for CV
        n_splits = min(3, groups_train.nunique())
        if n_splits < 2:
            # Fallback if too few groups in train_df
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=3)
            search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
            search.fit(X_train, y_train)
        else:
            gkf = GroupKFold(n_splits=n_splits)
            search = GridSearchCV(pipeline, param_grid, cv=gkf, scoring="f1_macro", n_jobs=-1)
            search.fit(X_train, y_train, groups=groups_train)

        best_model = search.best_estimator_
        
        # Evaluate
        classes = sorted(df["label"].unique().tolist())
        evaluator = Evaluator(best_model, X_test, y_test, classes)
        metrics = evaluator.evaluate()
        
        # Save artifact
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{model_type}"
        run_dir = self.dataset.data_dir.parent / "models" / "runs" / run_name
        
        artifact = ModelArtifact(run_dir)
        artifact.save(best_model, self.feature_names, classes, metrics)
        
        # Plot confusion matrix
        evaluator.plot_confusion_matrix(run_dir / "confusion_matrix.png")
        
        logger.info("Training complete. Model saved to %s", run_dir)
        
        # Sanity Checks
        train_pred = best_model.predict(X_train)
        from sklearn.metrics import accuracy_score
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = metrics["accuracy"]
        
        if test_acc == 1.0:
            logger.warning("Test accuracy is 1.0. This is suspiciously perfect (possible leakage).")
        if train_acc - test_acc > 0.15:
            logger.warning("Overfitting alert: Train accuracy (%.2f) >> Test accuracy (%.2f)", train_acc, test_acc)
            
        return run_dir

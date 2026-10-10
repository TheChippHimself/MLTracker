"""Model evaluation and report generation."""

import logging
from typing import Any

import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates trained models and generates reports."""

    def __init__(self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, classes: list[str]) -> None:
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.classes = classes
        
        self.y_pred = self.model.predict(self.X_test)

    def evaluate(self) -> dict[str, Any]:
        """Run all evaluations and return a metrics dictionary."""
        metrics: dict[str, Any] = {}
        
        metrics["accuracy"] = accuracy_score(self.y_test, self.y_pred)
        metrics["balanced_accuracy"] = balanced_accuracy_score(self.y_test, self.y_pred)
        metrics["macro_f1"] = f1_score(self.y_test, self.y_pred, average="macro", zero_division=0)
        
        # Classification report dictionary
        report = classification_report(self.y_test, self.y_pred, output_dict=True, zero_division=0)
        metrics["classification_report"] = report
        
        # Most confused pairs
        cm = confusion_matrix(self.y_test, self.y_pred, labels=self.classes)
        confusions = []
        for i, true_label in enumerate(self.classes):
            for j, pred_label in enumerate(self.classes):
                if i != j and cm[i, j] > 0:
                    confusions.append({
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "count": int(cm[i, j])
                    })
        
        # Sort by count descending
        confusions.sort(key=lambda x: x["count"], reverse=True)
        metrics["most_confused_pairs"] = confusions[:3]
        
        # Find weakest class
        class_f1s = []
        for cls in self.classes:
            if cls in report:
                class_f1s.append({"label": cls, "f1": report[cls]["f1-score"]})
                
        if class_f1s:
            class_f1s.sort(key=lambda x: x["f1"])
            metrics["weakest_class"] = class_f1s[0]
            
        return metrics

    def plot_confusion_matrix(self, out_path: str | Path) -> None:
        """Plot and save the confusion matrix."""
        cm = confusion_matrix(self.y_test, self.y_pred, labels=self.classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classes)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(str(out_path))
        plt.close(fig)

from __future__ import annotations

from sklearn.metrics import accuracy_score, classification_report, f1_score


def compute_classification_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def build_classification_report(y_true, y_pred, target_names: list[str]) -> str:
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)

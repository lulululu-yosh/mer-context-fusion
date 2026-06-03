from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    id2label: dict[int, str] | None = None,
) -> dict[str, Any]:
    labels = sorted(id2label.keys()) if id2label else sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    per_class = {}
    for index, label_id in enumerate(labels):
        label_name = id2label[label_id] if id2label else str(label_id)
        per_class[label_name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }

    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }

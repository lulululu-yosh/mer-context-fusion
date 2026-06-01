from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.training.trainer import DEFAULT_ID2EMOTION
from src.utils.io import ensure_dir, load_yaml, save_json


def get_id2emotion_from_config(config_path: str | Path | None) -> dict[int, str]:
    if config_path is None:
        return DEFAULT_ID2EMOTION

    cfg = load_yaml(config_path)

    if "labels" in cfg and "id2emotion" in cfg["labels"]:
        return {int(k): v for k, v in cfg["labels"]["id2emotion"].items()}

    return DEFAULT_ID2EMOTION


def plot_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    output_path: str | Path,
    normalize: bool = False,
) -> None:
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize="true" if normalize else None,
    )

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Text-only RoBERTa Confusion Matrix")
    plt.colorbar()

    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    fmt = ".2f" if normalize else "d"
    threshold = cm.max() / 2.0 if cm.size > 0 else 0.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=9,
            )

    plt.ylabel("Gold Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=300)
    plt.close()


def build_error_cases(pred_df: pd.DataFrame, max_cases_per_pair: int = 20) -> pd.DataFrame:
    wrong = pred_df[pred_df["gold"] != pred_df["pred"]].copy()

    if wrong.empty:
        return wrong

    wrong["confusion_pair"] = wrong["gold"] + " -> " + wrong["pred"]

    pred_confidence = []

    for _, row in wrong.iterrows():
        col = f"prob_{row['pred']}"
        pred_confidence.append(row[col] if col in wrong.columns else None)

    wrong["pred_confidence"] = pred_confidence

    wrong = wrong.sort_values(
        by=["confusion_pair", "pred_confidence"],
        ascending=[True, False],
    )

    wrong = wrong.groupby("confusion_pair", group_keys=False).head(max_cases_per_pair)

    preferred_cols = [
        "sample_id",
        "text",
        "gold",
        "pred",
        "confusion_pair",
        "pred_confidence",
    ]

    existing_preferred = [col for col in preferred_cols if col in wrong.columns]
    other_cols = [col for col in wrong.columns if col not in existing_preferred]

    return wrong[existing_preferred + other_cols].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp_dir",
        default="experiments/exp001_text_roberta_utt",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
    )
    parser.add_argument(
        "--config",
        default="configs/text/roberta_utt.yaml",
    )
    parser.add_argument(
        "--prediction_file",
        default=None,
        help="Default: <exp_dir>/dev_predictions_best.csv",
    )
    parser.add_argument(
        "--metrics_file",
        default=None,
        help="Default: <exp_dir>/best_metrics.json",
    )
    parser.add_argument(
        "--normalize_cm",
        action="store_true",
    )

    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    output_dir = Path(args.output_dir)

    prediction_file = (
        Path(args.prediction_file)
        if args.prediction_file
        else exp_dir / "dev_predictions_best.csv"
    )

    metrics_file = (
        Path(args.metrics_file)
        if args.metrics_file
        else exp_dir / "best_metrics.json"
    )

    if not prediction_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_file}")

    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    id2emotion = get_id2emotion_from_config(args.config)
    labels = [id2emotion[i] for i in sorted(id2emotion.keys())]

    pred_df = pd.read_csv(prediction_file)

    required_cols = {"sample_id", "text", "gold", "pred"}
    missing = required_cols - set(pred_df.columns)

    if missing:
        raise ValueError(
            f"Prediction file missing columns: {sorted(missing)}. "
            f"Available columns: {list(pred_df.columns)}"
        )

    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    predictions_dir = output_dir / "predictions"

    ensure_dir(figures_dir)
    ensure_dir(tables_dir)
    ensure_dir(predictions_dir)

    plot_confusion_matrix(
        y_true=pred_df["gold"].tolist(),
        y_pred=pred_df["pred"].tolist(),
        labels=labels,
        output_path=figures_dir / "confusion_matrix_text.png",
        normalize=args.normalize_cm,
    )

    metrics_df = pd.read_json(metrics_file, typ="series").to_frame().T
    metrics_df.insert(0, "model", "RoBERTa-text-only")
    metrics_df.to_csv(tables_dir / "text_only_metrics.csv", index=False)

    pred_df.to_csv(predictions_dir / "text_roberta_dev.csv", index=False)

    error_cases = build_error_cases(pred_df)
    error_cases.to_csv(tables_dir / "text_error_cases.csv", index=False)

    report_path = exp_dir / "classification_report.txt"
    if report_path.exists():
        report_text = report_path.read_text(encoding="utf-8")
        (tables_dir / "text_classification_report.txt").write_text(
            report_text,
            encoding="utf-8",
        )

    print("Text-only analysis completed.")
    print(f"Confusion matrix: {figures_dir / 'confusion_matrix_text.png'}")
    print(f"Metrics table: {tables_dir / 'text_only_metrics.csv'}")
    print(f"Predictions: {predictions_dir / 'text_roberta_dev.csv'}")
    print(f"Error cases: {tables_dir / 'text_error_cases.csv'}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import ensure_dir, load_yaml, read_jsonl


def analyze_split(jsonl_path: str, split: str) -> pd.DataFrame:
    rows = read_jsonl(jsonl_path)
    df = pd.DataFrame(rows)

    counts = (
        df["emotion"]
        .value_counts()
        .rename_axis("emotion")
        .reset_index(name="count")
        .sort_values("emotion")
    )
    counts["split"] = split
    counts["ratio"] = counts["count"] / counts["count"].sum()

    return counts[["split", "emotion", "count", "ratio"]]


def plot_distribution(counts: pd.DataFrame, output_path: str | Path) -> None:
    pivot = counts.pivot(index="emotion", columns="split", values="count").fillna(0)

    plt.figure(figsize=(10, 5))
    pivot.plot(kind="bar")
    plt.title("MELD Emotion Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.xticks(rotation=30)
    plt.tight_layout()

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/data/meld.yaml",
        help="Path to MELD data config.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    all_counts = pd.concat(
        [
            analyze_split(cfg["output"]["train_jsonl"], "train"),
            analyze_split(cfg["output"]["dev_jsonl"], "dev"),
            analyze_split(cfg["output"]["test_jsonl"], "test"),
        ],
        ignore_index=True,
    )

    table_dir = Path(cfg["analysis"]["table_dir"])
    figure_dir = Path(cfg["analysis"]["figure_dir"])
    ensure_dir(table_dir)
    ensure_dir(figure_dir)

    table_path = table_dir / "meld_label_distribution.csv"
    figure_path = figure_dir / "meld_label_distribution.png"

    all_counts.to_csv(table_path, index=False)
    plot_distribution(all_counts, figure_path)

    print(all_counts)
    print(f"\nSaved table to: {table_path}")
    print(f"Saved figure to: {figure_path}")


if __name__ == "__main__":
    main()
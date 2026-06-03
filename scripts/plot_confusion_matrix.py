from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.io import ensure_dir


def resolve_matrix_path(input_path: str) -> tuple[Path, str]:
    path = Path(input_path)
    if path.is_dir():
        return path / "confusion_matrix.csv", path.name
    return path, path.stem.replace("confusion_matrix_", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Experiment directory or confusion matrix CSV.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    matrix_path, exp_name = resolve_matrix_path(args.input)
    matrix = pd.read_csv(matrix_path, index_col=0)
    output = Path(args.output) if args.output else Path(
        f"outputs/figures/confusion_matrix_{exp_name}.png"
    )
    ensure_dir(output.parent)

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.ylabel("Gold")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"Saved confusion matrix figure to {output}")


if __name__ == "__main__":
    main()

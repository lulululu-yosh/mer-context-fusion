from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import load_yaml, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    rows = read_jsonl(cfg["paths"]["train_jsonl"])
    df = pd.DataFrame(rows)

    counts = df["emotion"].value_counts().sort_index()
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    counts.plot(kind="bar")
    plt.title("MELD Train Emotion Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution.png", dpi=200)

    print(counts)


if __name__ == "__main__":
    main()

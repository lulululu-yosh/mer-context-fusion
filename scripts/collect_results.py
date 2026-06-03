from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import ensure_dir, load_yaml


def load_metrics(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_experiment(exp_dir: Path) -> dict[str, Any] | None:
    metrics_path = exp_dir / "metrics.json"
    config_path = exp_dir / "config.yaml"
    if not metrics_path.exists() or not config_path.exists():
        return None

    metrics = load_metrics(metrics_path)
    cfg = load_yaml(config_path)
    modality = cfg.get("experiment", {}).get("modality", "")
    encoder = cfg.get("features", {}).get("model_name") or cfg.get("features", {}).get("encoder_name")
    if not encoder:
        encoder = cfg.get("model", {}).get("encoder_name", "")

    return {
        "exp_name": exp_dir.name,
        "modality": modality,
        "encoder": encoder,
        "accuracy": metrics.get("accuracy", ""),
        "macro_f1": metrics.get("macro_f1", ""),
        "weighted_f1": metrics.get("weighted_f1", ""),
        "best_epoch": metrics.get("epoch", ""),
        "seed": cfg.get("experiment", {}).get("seed", ""),
        "notes": cfg.get("experiment", {}).get("notes", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_dir", default="experiments")
    parser.add_argument("--output", default="outputs/tables/unimodal_results.csv")
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    rows = []
    for exp_dir in sorted(path for path in experiments_dir.iterdir() if path.is_dir()):
        row = read_experiment(exp_dir)
        if row:
            rows.append(row)

    output = Path(args.output)
    ensure_dir(output.parent)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved {len(rows)} experiment rows to {output}")


if __name__ == "__main__":
    main()

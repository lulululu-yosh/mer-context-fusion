from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features.media_paths import resolve_media_path
from src.utils.io import ensure_dir, load_yaml, read_jsonl


def analyze_split(cfg: dict, split: str) -> list[dict]:
    rows = read_jsonl(cfg["data"][f"{split}_file"])
    result = []
    for modality in ["audio", "video"]:
        present = 0
        missing = 0
        for row in rows:
            path = resolve_media_path(row, cfg, modality, split=split)
            if path is not None and Path(path).exists():
                present += 1
            else:
                missing += 1
        result.append(
            {
                "split": split,
                "modality": "visual" if modality == "video" else modality,
                "present": present,
                "missing": missing,
                "total": present + missing,
                "missing_rate": missing / max(1, present + missing),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/visual/resnet_frames.yaml")
    parser.add_argument("--output", default="outputs/tables/media_availability.csv")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    rows = []
    for split in ["train", "dev", "test"]:
        rows.extend(analyze_split(cfg, split))

    output = Path(args.output)
    ensure_dir(output.parent)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved media availability table to {output}")


if __name__ == "__main__":
    main()

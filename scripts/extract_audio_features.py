from __future__ import annotations

import argparse

from src.features.extract_audio_features import extract_audio_split
from src.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/audio/wav2vec2.yaml")
    parser.add_argument("--split", choices=["train", "dev", "test", "all"], default="all")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    splits = ["train", "dev", "test"] if args.split == "all" else [args.split]

    for split in splits:
        extract_audio_split(
            cfg=cfg,
            split=split,
            input_path=cfg["data"][f"{split}_file"],
            output_path=cfg["data"][f"{split}_features"],
        )


if __name__ == "__main__":
    main()

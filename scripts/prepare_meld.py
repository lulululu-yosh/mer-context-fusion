from __future__ import annotations

import argparse

from src.data.meld_reader import read_meld_csv
from src.utils.io import load_yaml, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    emotion2id = cfg["labels"]["emotion2id"]
    context_window = cfg["context"]["default_window"]

    train_rows = read_meld_csv(cfg["dataset"]["train_csv"], emotion2id, context_window=context_window)
    dev_rows = read_meld_csv(cfg["dataset"]["dev_csv"], emotion2id, context_window=context_window)
    test_rows = read_meld_csv(cfg["dataset"]["test_csv"], emotion2id, context_window=context_window)

    write_jsonl(train_rows, cfg["paths"]["train_jsonl"])
    write_jsonl(dev_rows, cfg["paths"]["dev_jsonl"])
    write_jsonl(test_rows, cfg["paths"]["test_jsonl"])

    print(f"Saved {len(train_rows)} train, {len(dev_rows)} dev, {len(test_rows)} test samples.")


if __name__ == "__main__":
    main()

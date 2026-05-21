from __future__ import annotations

import argparse
from pprint import pprint

from src.data.meld_reader import read_meld_csv
from src.data.sanity_check import run_basic_checks
from src.utils.io import load_yaml, write_jsonl


def process_split(
    split: str,
    csv_path: str,
    output_path: str,
    emotion2id: dict[str, int],
    context_cfg: dict,
) -> None:
    rows = read_meld_csv(
        csv_path=csv_path,
        split=split,
        emotion2id=emotion2id,
        window_size=int(context_cfg["window_size"]),
        include_current=bool(context_cfg["include_current"]),
        speaker_prefix=bool(context_cfg["speaker_prefix"]),
        sep_token=str(context_cfg["sep_token"]),
    )

    summary = run_basic_checks(rows, num_labels=len(emotion2id))
    write_jsonl(rows, output_path)

    print(f"\n[{split}] saved to {output_path}")
    pprint(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/data/meld.yaml",
        help="Path to MELD data config.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    emotion2id = cfg["labels"]["emotion2id"]
    context_cfg = cfg["context"]

    process_split(
        split="train",
        csv_path=cfg["dataset"]["train_csv"],
        output_path=cfg["output"]["train_jsonl"],
        emotion2id=emotion2id,
        context_cfg=context_cfg,
    )

    process_split(
        split="dev",
        csv_path=cfg["dataset"]["dev_csv"],
        output_path=cfg["output"]["dev_jsonl"],
        emotion2id=emotion2id,
        context_cfg=context_cfg,
    )

    process_split(
        split="test",
        csv_path=cfg["dataset"]["test_csv"],
        output_path=cfg["output"]["test_jsonl"],
        emotion2id=emotion2id,
        context_cfg=context_cfg,
    )


if __name__ == "__main__":
    main()
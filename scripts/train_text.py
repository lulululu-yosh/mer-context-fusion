from __future__ import annotations

import argparse

from transformers import AutoTokenizer

from src.data.meld_dataset import MELDTextDataset
from src.models.text_classifier import TextEmotionClassifier
from src.training.trainer import DEFAULT_ID2EMOTION, TextTrainer
from src.utils.io import load_yaml
from src.utils.seed import set_seed


def get_id2emotion(cfg: dict) -> dict[int, str]:
    if "labels" in cfg and "id2emotion" in cfg["labels"]:
        return {int(k): v for k, v in cfg["labels"]["id2emotion"].items()}

    return DEFAULT_ID2EMOTION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/text/roberta_utt.yaml",
        help="Path to text baseline config.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["experiment"].get("seed", 42)))

    id2emotion = get_id2emotion(cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])

    train_dataset = MELDTextDataset(
        jsonl_path=cfg["data"]["train_file"],
        tokenizer=tokenizer,
        text_field=cfg["data"].get("text_field", "text"),
        label_field=cfg["data"].get("label_field", "label"),
        max_length=int(cfg["training"].get("max_length", 128)),
    )

    dev_dataset = MELDTextDataset(
        jsonl_path=cfg["data"]["dev_file"],
        tokenizer=tokenizer,
        text_field=cfg["data"].get("text_field", "text"),
        label_field=cfg["data"].get("label_field", "label"),
        max_length=int(cfg["training"].get("max_length", 128)),
    )

    model = TextEmotionClassifier(
        encoder_name=cfg["model"]["encoder_name"],
        num_labels=int(cfg["model"].get("num_labels", 7)),
        dropout=float(cfg["model"].get("dropout", 0.1)),
    )

    trainer = TextTrainer(
        model=model,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        config=cfg,
        config_path=args.config,
        id2emotion=id2emotion,
    )

    trainer.train()


if __name__ == "__main__":
    main()
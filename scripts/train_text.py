from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from transformers import AutoTokenizer

from src.data.meld_dataset import MELDTextDataset
from src.models.text_classifier import TextClassifier
from src.training.trainer import SimpleTrainer
from src.utils.io import load_yaml
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["experiment"]["seed"]))

    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, output_dir / "config.yaml")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])

    train_ds = MELDTextDataset(
        cfg["data"]["train_file"],
        tokenizer=tokenizer,
        text_field=cfg["data"]["text_field"],
        label_field=cfg["data"]["label_field"],
        max_length=int(cfg["training"]["max_length"]),
    )
    dev_ds = MELDTextDataset(
        cfg["data"]["dev_file"],
        tokenizer=tokenizer,
        text_field=cfg["data"]["text_field"],
        label_field=cfg["data"]["label_field"],
        max_length=int(cfg["training"]["max_length"]),
    )

    model = TextClassifier(
        encoder_name=cfg["model"]["encoder_name"],
        num_labels=int(cfg["model"]["num_labels"]),
        dropout=float(cfg["model"]["dropout"]),
    )

    trainer = SimpleTrainer(model, train_ds, dev_ds, cfg)
    metrics = trainer.train()
    print(metrics)


if __name__ == "__main__":
    main()

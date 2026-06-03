from __future__ import annotations

import argparse

import torch

from src.data.feature_dataset import CachedFeatureDataset
from src.models.unimodal_mlp import FeatureMLPClassifier
from src.training.trainer import DEFAULT_ID2EMOTION, TextTrainer
from src.utils.io import load_yaml
from src.utils.seed import set_seed


def get_id2emotion(cfg: dict) -> dict[int, str]:
    if "labels" in cfg and "id2emotion" in cfg["labels"]:
        return {int(k): v for k, v in cfg["labels"]["id2emotion"].items()}
    return DEFAULT_ID2EMOTION


def compute_class_weights(dataset: CachedFeatureDataset, num_labels: int) -> torch.Tensor:
    counts = torch.zeros(num_labels, dtype=torch.float32)
    for item in dataset.items:
        counts[int(item["label"])] += 1.0
    weights = counts.sum() / counts.clamp(min=1.0)
    return weights / weights.mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/visual/resnet_frames.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["experiment"].get("seed", 42)))
    id2emotion = get_id2emotion(cfg)

    train_dataset = CachedFeatureDataset(cfg["data"]["train_features"])
    dev_dataset = CachedFeatureDataset(cfg["data"]["dev_features"])

    num_labels = int(cfg["model"].get("num_labels", 7))
    class_weights = None
    if bool(cfg["training"].get("use_class_weights", False)):
        class_weights = compute_class_weights(train_dataset, num_labels)

    model = FeatureMLPClassifier(
        input_dim=train_dataset.input_dim,
        hidden_dim=int(cfg["model"].get("hidden_dim", 256)),
        num_labels=num_labels,
        dropout=float(cfg["model"].get("dropout", 0.1)),
        class_weights=class_weights,
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

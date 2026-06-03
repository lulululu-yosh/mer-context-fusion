from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.feature_dataset import CachedFeatureDataset
from src.evaluation.metrics import compute_metrics
from src.models.unimodal_mlp import FeatureMLPClassifier
from src.training.trainer import DEFAULT_ID2EMOTION
from src.utils.io import ensure_dir, load_yaml, save_json


def batch_value(batch: dict[str, Any], key: str, index: int, default: Any = "") -> Any:
    if key not in batch:
        return default
    value = batch[key]
    if isinstance(value, torch.Tensor):
        item = value[index]
        if item.ndim == 0:
            return item.item()
        return item.detach().cpu().tolist()
    return value[index]


def get_id2emotion(cfg: dict) -> dict[int, str]:
    if "labels" in cfg and "id2emotion" in cfg["labels"]:
        return {int(k): v for k, v in cfg["labels"]["id2emotion"].items()}
    return DEFAULT_ID2EMOTION


def get_feature_path(cfg: dict[str, Any], split: str) -> str:
    return cfg["data"][f"{split}_features"]


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    id2emotion: dict[int, str],
) -> tuple[dict[str, Any], str, pd.DataFrame]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    records: list[dict[str, Any]] = []

    for batch in tqdm(dataloader, desc="evaluating"):
        features = batch["features"].to(device)
        labels_tensor = batch["labels"].to(device)
        outputs = model(features=features, labels=labels_tensor)
        probs = torch.softmax(outputs["logits"], dim=-1)
        preds = torch.argmax(probs, dim=-1)

        labels = labels_tensor.cpu().tolist()
        preds_list = preds.cpu().tolist()
        probs_list = probs.cpu().tolist()
        y_true.extend(labels)
        y_pred.extend(preds_list)

        for i, gold_id in enumerate(labels):
            pred_id = int(preds_list[i])
            record = {
                "sample_id": batch_value(batch, "sample_id", i),
                "dialogue_id": batch_value(batch, "dialogue_id", i),
                "utterance_id": batch_value(batch, "utterance_id", i),
                "speaker": batch_value(batch, "speaker", i),
                "text": batch_value(batch, "text", i),
                "gold": id2emotion[int(gold_id)],
                "pred": id2emotion[pred_id],
            }
            for key, value in batch.items():
                if key.startswith("missing_"):
                    record[key] = bool(batch_value(batch, key, i, False))
            for label_id, emotion in id2emotion.items():
                record[f"prob_{emotion}"] = probs_list[i][label_id]
            records.append(record)

    sorted_ids = sorted(id2emotion.keys())
    report = classification_report(
        y_true,
        y_pred,
        labels=sorted_ids,
        target_names=[id2emotion[i] for i in sorted_ids],
        digits=4,
        zero_division=0,
    )
    return compute_metrics(y_true, y_pred, id2label=id2emotion), report, pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    id2emotion = get_id2emotion(cfg)
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg["experiment"]["output_dir"])
    ensure_dir(output_dir)

    dataset = CachedFeatureDataset(get_feature_path(cfg, args.split))
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg["training"].get("batch_size", 64)),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 0)),
    )

    model = FeatureMLPClassifier(
        input_dim=dataset.input_dim,
        hidden_dim=int(cfg["model"].get("hidden_dim", 256)),
        num_labels=int(cfg["model"].get("num_labels", 7)),
        dropout=float(cfg["model"].get("dropout", 0.1)),
    )
    checkpoint = Path(args.checkpoint) if args.checkpoint else output_dir / "best_model.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    metrics, report, predictions = evaluate(model, dataloader, device, id2emotion)
    labels = [id2emotion[i] for i in sorted(id2emotion.keys())]
    matrix = confusion_matrix(predictions["gold"], predictions["pred"], labels=labels)

    save_json(metrics, output_dir / f"{args.split}_metrics.json")
    with open(output_dir / f"{args.split}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    predictions.to_csv(output_dir / f"{args.split}_predictions.csv", index=False)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        output_dir / f"{args.split}_confusion_matrix.csv"
    )
    print(metrics)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data.meld_dataset import MELDTextDataset
from src.evaluation.metrics import compute_metrics
from src.models.text_classifier import TextEmotionClassifier
from src.training.trainer import DEFAULT_ID2EMOTION
from src.utils.io import ensure_dir, load_yaml, save_json


def get_id2emotion(cfg: dict) -> dict[int, str]:
    if "labels" in cfg and "id2emotion" in cfg["labels"]:
        return {int(k): v for k, v in cfg["labels"]["id2emotion"].items()}

    return DEFAULT_ID2EMOTION


def get_split_file(cfg: dict[str, Any], split: str) -> str:
    if split == "train":
        return cfg["data"]["train_file"]
    if split == "dev":
        return cfg["data"]["dev_file"]
    if split == "test":
        return cfg["data"]["test_file"]

    raise ValueError(f"Unsupported split: {split}")


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask", "labels"}
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    id2emotion: dict[int, str],
) -> tuple[dict[str, float], str, pd.DataFrame]:
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    records: list[dict[str, Any]] = []

    for batch in tqdm(dataloader, desc="evaluating"):
        model_inputs = move_batch_to_device(batch, device)

        outputs = model(**model_inputs)
        logits = outputs["logits"]

        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        labels = model_inputs["labels"].detach().cpu().tolist()
        preds_list = preds.detach().cpu().tolist()
        probs_list = probs.detach().cpu().tolist()

        y_true.extend(labels)
        y_pred.extend(preds_list)

        batch_size = len(labels)

        for i in range(batch_size):
            gold_id = int(labels[i])
            pred_id = int(preds_list[i])

            record = {
                "sample_id": batch["sample_id"][i],
                "text": batch["text"][i],
                "gold": id2emotion[gold_id],
                "pred": id2emotion[pred_id],
                "gold_from_data": batch["emotion"][i],
            }

            for label_id, emotion in id2emotion.items():
                record[f"prob_{emotion}"] = probs_list[i][label_id]

            records.append(record)

    metrics = compute_metrics(y_true, y_pred, id2label=id2emotion)

    sorted_ids = sorted(id2emotion.keys())

    report = classification_report(
        y_true,
        y_pred,
        labels=sorted_ids,
        target_names=[id2emotion[i] for i in sorted_ids],
        digits=4,
        zero_division=0,
    )

    return metrics, report, pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="configs/text/roberta_utt.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--output_dir", default=None)

    args = parser.parse_args()

    cfg = load_yaml(args.config)
    id2emotion = get_id2emotion(cfg)

    exp_output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir = Path(args.output_dir) if args.output_dir else exp_output_dir
    ensure_dir(output_dir)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else exp_output_dir / "best_model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    split_file = get_split_file(cfg, args.split)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Split: {args.split}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data file: {split_file}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])

    dataset = MELDTextDataset(
        jsonl_path=split_file,
        tokenizer=tokenizer,
        text_field=cfg["data"].get("text_field", "text"),
        label_field=cfg["data"].get("label_field", "label"),
        max_length=int(cfg["training"].get("max_length", 128)),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg["training"].get("batch_size", 16)),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 0)),
    )

    model = TextEmotionClassifier(
        encoder_name=cfg["model"]["encoder_name"],
        num_labels=int(cfg["model"].get("num_labels", 7)),
        dropout=float(cfg["model"].get("dropout", 0.1)),
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)

    metrics, report, predictions = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        id2emotion=id2emotion,
    )

    save_json(metrics, output_dir / f"{args.split}_metrics.json")

    with open(output_dir / f"{args.split}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    predictions.to_csv(output_dir / f"{args.split}_predictions.csv", index=False)

    print("\nEvaluation finished.")
    print(metrics)


if __name__ == "__main__":
    main()

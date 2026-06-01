from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.evaluation.metrics import compute_metrics
from src.utils.io import ensure_dir, save_json


DEFAULT_ID2EMOTION = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "joy",
    4: "neutral",
    5: "sadness",
    6: "surprise",
}


class TextTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset,
        dev_dataset,
        config: dict[str, Any],
        config_path: str | Path,
        id2emotion: dict[int, str] | None = None,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.config = config
        self.config_path = Path(config_path)
        self.id2emotion = id2emotion or DEFAULT_ID2EMOTION

        self.output_dir = Path(config["experiment"]["output_dir"])
        ensure_dir(self.output_dir)

        shutil.copy(self.config_path, self.output_dir / "config.yaml")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.best_metric_name = config["training"].get("save_best_by", "macro_f1")
        self.best_score = -1.0
        self.history: list[dict[str, Any]] = []

        print(f"Using device: {self.device}")

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {
            key: value.to(self.device)
            for key, value in batch.items()
            if key in {"input_ids", "attention_mask", "labels"}
        }

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        cfg = self.config["training"]

        return DataLoader(
            dataset,
            batch_size=int(cfg["batch_size"]),
            shuffle=shuffle,
            num_workers=int(cfg.get("num_workers", 0)),
        )

    def train(self) -> dict[str, Any]:
        cfg = self.config["training"]

        train_loader = self._make_loader(self.train_dataset, shuffle=True)
        dev_loader = self._make_loader(self.dev_dataset, shuffle=False)

        optimizer = AdamW(
            self.model.parameters(),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
        )

        total_steps = len(train_loader) * int(cfg["epochs"])
        warmup_ratio = float(cfg.get("warmup_ratio", 0.1))

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(warmup_ratio * total_steps),
            num_training_steps=total_steps,
        )

        for epoch in range(1, int(cfg["epochs"]) + 1):
            train_loss = self._train_one_epoch(
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )

            dev_metrics, dev_report, dev_predictions = self.evaluate(dev_loader)

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                **dev_metrics,
            }

            self.history.append(epoch_metrics)

            save_json(epoch_metrics, self.output_dir / "last_metrics.json")
            dev_predictions.to_csv(self.output_dir / "dev_predictions_last.csv", index=False)
            pd.DataFrame(self.history).to_csv(self.output_dir / "metrics_history.csv", index=False)

            print(f"\nEpoch {epoch}")
            print(f"train_loss: {train_loss:.4f}")
            print(dev_metrics)
            print(dev_report)

            current_score = float(dev_metrics[self.best_metric_name])

            if current_score > self.best_score:
                self.best_score = current_score
                self._save_best(epoch_metrics, dev_report, dev_predictions)
                print(f"Saved best model with {self.best_metric_name}={self.best_score:.4f}")

        print(f"\nTraining finished. Best {self.best_metric_name}: {self.best_score:.4f}")

        return {
            "best_metric": self.best_metric_name,
            "best_score": self.best_score,
        }

    def _train_one_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch: int,
    ) -> float:
        self.model.train()
        total_loss = 0.0

        progress = tqdm(train_loader, desc=f"epoch {epoch}")

        for batch in progress:
            model_inputs = self._move_batch_to_device(batch)

            optimizer.zero_grad()
            outputs = self.model(**model_inputs)
            loss = outputs["loss"]

            loss.backward()

            max_grad_norm = float(self.config["training"].get("max_grad_norm", 1.0))
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item())
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(1, len(train_loader))

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> tuple[dict[str, float], str, pd.DataFrame]:
        self.model.eval()

        y_true: list[int] = []
        y_pred: list[int] = []
        records: list[dict[str, Any]] = []

        for batch in tqdm(dataloader, desc="evaluating", leave=False):
            model_inputs = self._move_batch_to_device(batch)

            outputs = self.model(**model_inputs)
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
                    "gold": self.id2emotion[gold_id],
                    "pred": self.id2emotion[pred_id],
                    "gold_from_data": batch["emotion"][i],
                }

                for label_id, emotion in self.id2emotion.items():
                    record[f"prob_{emotion}"] = probs_list[i][label_id]

                records.append(record)

        metrics = compute_metrics(y_true, y_pred)

        sorted_ids = sorted(self.id2emotion.keys())

        report = classification_report(
            y_true,
            y_pred,
            labels=sorted_ids,
            target_names=[self.id2emotion[i] for i in sorted_ids],
            digits=4,
            zero_division=0,
        )

        predictions = pd.DataFrame(records)

        return metrics, report, predictions

    def _save_best(
        self,
        metrics: dict[str, Any],
        report: str,
        predictions: pd.DataFrame,
    ) -> None:
        torch.save(self.model.state_dict(), self.output_dir / "best_model.pt")

        save_json(metrics, self.output_dir / "best_metrics.json")

        with open(self.output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

        predictions.to_csv(self.output_dir / "dev_predictions_best.csv", index=False)
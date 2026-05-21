from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import compute_classification_metrics
from src.utils.io import save_json


class SimpleTrainer:
    def __init__(self, model, train_dataset, dev_dataset, config: dict, device: str | None = None):
        self.model = model
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self) -> dict:
        training_cfg = self.config["training"]
        output_dir = Path(self.config["experiment"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=training_cfg["batch_size"],
            shuffle=True,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(training_cfg["lr"]),
            weight_decay=float(training_cfg.get("weight_decay", 0.0)),
        )

        for epoch in range(int(training_cfg["epochs"])):
            self.model.train()
            total_loss = 0.0
            for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}"):
                batch = {k: v.to(self.device) for k, v in batch.items() if k != "sample_id"}
                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs["loss"]
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            metrics = self.evaluate()
            metrics["train_loss"] = total_loss / max(1, len(train_loader))
            save_json(metrics, output_dir / "metrics.json")

        torch.save(self.model.state_dict(), output_dir / "model.pt")
        return metrics

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        loader = DataLoader(
            self.dev_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
        )

        y_true, y_pred = [], []
        for batch in loader:
            labels = batch["labels"].numpy().tolist()
            model_inputs = {k: v.to(self.device) for k, v in batch.items() if k not in {"labels", "sample_id"}}
            outputs = self.model(**model_inputs)
            preds = outputs["logits"].argmax(dim=-1).cpu().numpy().tolist()
            y_true.extend(labels)
            y_pred.extend(preds)

        return compute_classification_metrics(y_true, y_pred)

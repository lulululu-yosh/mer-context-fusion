from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.utils.io import read_jsonl


class MELDTextDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer,
        text_field: str = "text",
        label_field: str = "label",
        max_length: int = 128,
    ) -> None:
        self.rows = read_jsonl(jsonl_path)
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.label_field = label_field
        self.max_length = max_length

        if len(self.rows) == 0:
            raise ValueError(f"No samples found in {jsonl_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        text = str(row.get(self.text_field, ""))

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row[self.label_field]), dtype=torch.long),
            "sample_id": row.get("sample_id", f"sample_{index}"),
            "text": text,
            "emotion": row.get("emotion", ""),
        }
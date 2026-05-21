from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.utils.io import read_jsonl


class MELDTextDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer, text_field: str = "text", label_field: str = "label", max_length: int = 128):
        self.rows = read_jsonl(path)
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.label_field = label_field
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        encoded = self.tokenizer(
            row[self.text_field],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(row[self.label_field], dtype=torch.long)
        item["sample_id"] = row["sample_id"]
        return item

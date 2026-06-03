from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class CachedFeatureDataset(Dataset):
    """Dataset for cached utterance-level modality features."""

    def __init__(
        self,
        feature_path: str | Path,
        label_field: str = "label",
        feature_field: str = "embedding",
    ) -> None:
        self.feature_path = Path(feature_path)
        if not self.feature_path.exists():
            raise FileNotFoundError(f"Feature cache not found: {self.feature_path}")

        cache = torch.load(self.feature_path, map_location="cpu")
        if isinstance(cache, dict) and "items" in cache:
            self.items = cache["items"]
            self.metadata = {k: v for k, v in cache.items() if k != "items"}
        elif isinstance(cache, list):
            self.items = cache
            self.metadata = {}
        else:
            raise ValueError(
                f"Unsupported cache format in {self.feature_path}. Expected dict with items or list."
            )

        if len(self.items) == 0:
            raise ValueError(f"No feature rows found in {self.feature_path}")

        self.label_field = label_field
        self.feature_field = feature_field
        first_embedding = self.items[0][self.feature_field]
        self.input_dim = int(torch.as_tensor(first_embedding).numel())

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.items[index]
        features = torch.as_tensor(row[self.feature_field], dtype=torch.float32).view(-1)

        sample = {
            "features": features,
            "labels": torch.tensor(int(row[self.label_field]), dtype=torch.long),
            "sample_id": row.get("sample_id", f"sample_{index}"),
            "dialogue_id": row.get("dialogue_id", ""),
            "utterance_id": row.get("utterance_id", ""),
            "speaker": row.get("speaker", ""),
            "text": row.get("text", ""),
            "emotion": row.get("emotion", ""),
        }

        for key, value in row.items():
            if key.startswith("missing_") or key in {"num_frames_used"}:
                sample[key] = value

        return sample

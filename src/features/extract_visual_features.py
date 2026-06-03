from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torchvision.io import read_video
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
from tqdm import tqdm

from src.features.media_paths import metadata_from_row, resolve_media_path
from src.utils.io import ensure_dir, read_jsonl


def build_resnet_encoder(name: str) -> tuple[torch.nn.Module, Any, int]:
    normalized = name.lower()
    if normalized == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        embedding_dim = int(model.fc.in_features)
    elif normalized == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        embedding_dim = int(model.fc.in_features)
    else:
        raise ValueError(f"Unsupported visual encoder: {name}")

    model.fc = torch.nn.Identity()
    return model, weights.transforms(), embedding_dim


def _sample_frames(video_path: Path, num_frames: int) -> torch.Tensor:
    video, _, _ = read_video(str(video_path), pts_unit="sec", output_format="TCHW")
    if video.numel() == 0 or video.shape[0] == 0:
        raise ValueError("empty video")

    frame_count = video.shape[0]
    indices = torch.linspace(0, frame_count - 1, steps=min(num_frames, frame_count)).long()
    frames = video[indices].float() / 255.0
    return frames


@torch.no_grad()
def extract_visual_split(
    cfg: dict[str, Any],
    split: str,
    input_path: str | Path,
    output_path: str | Path,
) -> None:
    rows = read_jsonl(input_path)
    feature_cfg = cfg["features"]
    encoder_name = feature_cfg.get("encoder_name", "resnet50")
    num_frames = int(feature_cfg.get("num_frames", 8))
    batch_size = int(feature_cfg.get("batch_size", 16))
    missing_policy = feature_cfg.get("missing_policy", "zero")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess, embedding_dim = build_resnet_encoder(encoder_name)
    model = model.to(device)
    model.eval()

    items: list[dict[str, Any]] = []
    missing_log: list[str] = []

    for row in tqdm(rows, desc=f"visual {split}"):
        record = metadata_from_row(row)
        record["missing_visual"] = False
        record["num_frames_used"] = 0
        media_path = resolve_media_path(row, cfg, "video", split=split)

        try:
            if media_path is None or not media_path.exists():
                raise FileNotFoundError(str(media_path) if media_path else "no video path")
            frames = _sample_frames(media_path, num_frames)
        except Exception as exc:
            missing_log.append(f"{record['sample_id']}\t{exc}")
            if missing_policy == "skip":
                continue
            record["missing_visual"] = True
            record["embedding"] = torch.zeros(embedding_dim)
            items.append(record)
            continue

        embeddings: list[torch.Tensor] = []
        for start in range(0, frames.shape[0], batch_size):
            batch = frames[start : start + batch_size]
            batch = preprocess(batch).to(device)
            embeddings.append(model(batch).cpu())

        pooled = torch.cat(embeddings, dim=0).mean(dim=0)
        record["embedding"] = pooled
        record["num_frames_used"] = int(frames.shape[0])
        items.append(record)

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    torch.save(
        {
            "split": split,
            "modality": "visual",
            "encoder": encoder_name,
            "embedding_dim": embedding_dim,
            "items": items,
        },
        output_path,
    )

    log_path = output_path.with_suffix(".missing.tsv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("sample_id\terror\n")
        for line in missing_log:
            f.write(line + "\n")

    print(f"Saved {len(items)} visual features to {output_path}")
    print(f"Logged {len(missing_log)} missing/corrupt visual samples to {log_path}")

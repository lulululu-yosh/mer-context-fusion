from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel

from src.features.media_paths import metadata_from_row, resolve_media_path
from src.utils.io import ensure_dir, read_jsonl


def _load_audio(
    path: Path,
    sample_rate: int,
    max_duration_seconds: float | None,
) -> torch.Tensor:
    waveform, original_rate = torchaudio.load(path)
    waveform = waveform.mean(dim=0)

    if original_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, original_rate, sample_rate)

    if max_duration_seconds:
        max_length = int(sample_rate * max_duration_seconds)
        waveform = waveform[:max_length]

    return waveform


def _mean_pool_hidden(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    masked_hidden = last_hidden_state * mask
    lengths = mask.sum(dim=1).clamp(min=1.0)
    return masked_hidden.sum(dim=1) / lengths


@torch.no_grad()
def extract_audio_split(
    cfg: dict[str, Any],
    split: str,
    input_path: str | Path,
    output_path: str | Path,
) -> None:
    rows = read_jsonl(input_path)
    feature_cfg = cfg["features"]
    model_name = feature_cfg["model_name"]
    sample_rate = int(feature_cfg.get("sample_rate", 16000))
    batch_size = int(feature_cfg.get("batch_size", 8))
    max_duration = feature_cfg.get("max_duration_seconds")
    max_duration = float(max_duration) if max_duration else None
    missing_policy = feature_cfg.get("missing_policy", "zero")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    hidden_size = int(model.config.hidden_size)

    items: list[dict[str, Any]] = []
    pending_waveforms: list[torch.Tensor] = []
    pending_records: list[dict[str, Any]] = []

    def flush() -> None:
        if not pending_waveforms:
            return

        inputs = processor(
            [waveform.numpy() for waveform in pending_waveforms],
            sampling_rate=sample_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)
        embeddings = _mean_pool_hidden(outputs.last_hidden_state, inputs["attention_mask"])

        for record, embedding in zip(pending_records, embeddings.cpu(), strict=True):
            record["embedding"] = embedding
            items.append(record)

        pending_waveforms.clear()
        pending_records.clear()

    missing_log: list[str] = []

    for row in tqdm(rows, desc=f"audio {split}"):
        record = metadata_from_row(row)
        record["missing_audio"] = False
        media_path = resolve_media_path(row, cfg, "audio", split=split)

        try:
            if media_path is None or not media_path.exists():
                raise FileNotFoundError(str(media_path) if media_path else "no audio path")
            waveform = _load_audio(media_path, sample_rate, max_duration)
            if waveform.numel() == 0:
                raise ValueError("empty waveform")
        except Exception as exc:
            missing_log.append(f"{record['sample_id']}\t{exc}")
            if missing_policy == "skip":
                continue
            record["missing_audio"] = True
            record["embedding"] = torch.zeros(hidden_size)
            items.append(record)
            continue

        pending_waveforms.append(waveform.cpu())
        pending_records.append(record)
        if len(pending_waveforms) >= batch_size:
            flush()

    flush()

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    torch.save(
        {
            "split": split,
            "modality": "audio",
            "encoder": model_name,
            "embedding_dim": hidden_size,
            "items": items,
        },
        output_path,
    )

    log_path = output_path.with_suffix(".missing.tsv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("sample_id\terror\n")
        for line in missing_log:
            f.write(line + "\n")

    print(f"Saved {len(items)} audio features to {output_path}")
    print(f"Logged {len(missing_log)} missing/corrupt audio samples to {log_path}")

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_media_path(
    row: dict[str, Any],
    cfg: dict[str, Any],
    modality: str,
    split: str | None = None,
) -> Path | None:
    field_name = f"{modality}_path"
    if row.get(field_name):
        return Path(str(row[field_name]))

    path_cfg = cfg.get("paths", {})
    root = Path(path_cfg.get("media_root", ""))
    pattern = path_cfg.get(f"{modality}_pattern")
    if not pattern:
        return None

    values = {key: "" if value is None else value for key, value in row.items()}
    if split is not None:
        values["split"] = split
    try:
        relative_path = pattern.format(**values)
    except KeyError as exc:
        raise KeyError(f"Unknown placeholder in {modality}_pattern: {exc}") from exc

    return root / relative_path


def metadata_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row.get("sample_id"),
        "dialogue_id": row.get("dialogue_id", ""),
        "utterance_id": row.get("utterance_id", ""),
        "speaker": row.get("speaker", ""),
        "text": row.get("text", ""),
        "emotion": row.get("emotion", ""),
        "label": int(row["label"]),
    }

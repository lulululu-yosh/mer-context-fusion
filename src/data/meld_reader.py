from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.build_context import build_context_text, normalize_text


MELD_COLUMN_MAP = {
    "Utterance": "text",
    "Speaker": "speaker",
    "Emotion": "emotion",
    "Sentiment": "sentiment",
    "Dialogue_ID": "dialogue_id",
    "Utterance_ID": "utterance_id",
    "Sr No.": "sr_no",
    "Season": "season",
    "Episode": "episode",
    "StartTime": "start_time",
    "EndTime": "end_time",
}


REQUIRED_COLUMNS = {
    "text",
    "speaker",
    "emotion",
    "sentiment",
    "dialogue_id",
    "utterance_id",
}


def standardize_meld_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename original MELD columns into project-standard names."""
    df = df.rename(columns={k: v for k, v in MELD_COLUMN_MAP.items() if k in df.columns})
    return df


def validate_required_columns(df: pd.DataFrame, csv_path: str | Path) -> None:
    """Ensure the CSV contains required fields."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )


def normalize_emotion(value: Any) -> str:
    """Normalize emotion label string."""
    return normalize_text(value).lower()


def build_sample_id(dialogue_id: int, utterance_id: int) -> str:
    """Build stable utterance-level sample ID."""
    return f"dia{dialogue_id}_utt{utterance_id}"


def read_meld_csv(
    csv_path: str | Path,
    split: str,
    emotion2id: dict[str, int],
    window_size: int = 3,
    include_current: bool = True,
    speaker_prefix: bool = True,
    sep_token: str = " [SEP] ",
) -> list[dict[str, Any]]:
    """
    Read one MELD split CSV and convert it into standardized utterance samples.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"MELD CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = standardize_meld_columns(df)
    validate_required_columns(df, csv_path)

    df["emotion"] = df["emotion"].apply(normalize_emotion)
    df["text"] = df["text"].apply(normalize_text)
    df["speaker"] = df["speaker"].apply(normalize_text)
    df["sentiment"] = df["sentiment"].apply(lambda x: normalize_text(x).lower())

    unknown_emotions = sorted(set(df["emotion"]) - set(emotion2id.keys()))
    if unknown_emotions:
        raise ValueError(f"Unknown emotion labels in {csv_path}: {unknown_emotions}")

    df["dialogue_id"] = df["dialogue_id"].astype(int)
    df["utterance_id"] = df["utterance_id"].astype(int)

    df = df.sort_values(["dialogue_id", "utterance_id"]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []

    for dialogue_id, group in df.groupby("dialogue_id", sort=False):
        dialogue_rows = group.to_dict("records")

        for current_index, row in enumerate(dialogue_rows):
            utterance_id = int(row["utterance_id"])
            emotion = row["emotion"]

            sample = {
                "sample_id": build_sample_id(int(dialogue_id), utterance_id),
                "split": split,
                "dialogue_id": int(dialogue_id),
                "utterance_id": utterance_id,
                "speaker": row["speaker"],
                "text": row["text"],
                "context_text": build_context_text(
                    dialogue_rows=dialogue_rows,
                    current_index=current_index,
                    window_size=window_size,
                    include_current=include_current,
                    speaker_prefix=speaker_prefix,
                    sep_token=sep_token,
                ),
                "emotion": emotion,
                "label": int(emotion2id[emotion]),
                "sentiment": row["sentiment"],
            }

            rows.append(sample)

    return rows
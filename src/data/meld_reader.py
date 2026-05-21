from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.build_context import build_context_text


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common MELD CSV column names into project fields."""
    rename_map = {
        "Dialogue_ID": "dialogue_id",
        "Utterance_ID": "utterance_id",
        "Speaker": "speaker",
        "Utterance": "text",
        "Emotion": "emotion",
        "Sentiment": "sentiment",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def read_meld_csv(csv_path: str | Path, emotion2id: dict[str, int], context_window: int = 3) -> list[dict]:
    df = pd.read_csv(csv_path)
    df = _standardize_columns(df)
    df = df.sort_values(["dialogue_id", "utterance_id"]).reset_index(drop=True)

    rows: list[dict] = []
    for _, group in df.groupby("dialogue_id", sort=False):
        dialogue = group.to_dict("records")
        for i, row in enumerate(dialogue):
            emotion = str(row["emotion"]).lower()
            sample = {
                "sample_id": f"dia{row['dialogue_id']}_utt{row['utterance_id']}",
                "dialogue_id": int(row["dialogue_id"]),
                "utterance_id": int(row["utterance_id"]),
                "speaker": str(row.get("speaker", "")),
                "text": str(row.get("text", "")),
                "context_text": build_context_text(dialogue, i, window=context_window, speaker_prefix=True),
                "emotion": emotion,
                "label": emotion2id[emotion],
                "sentiment": str(row.get("sentiment", "")).lower(),
            }
            rows.append(sample)

    return rows

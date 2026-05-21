from __future__ import annotations

from collections import Counter
from typing import Any


def check_sample_ids_unique(rows: list[dict[str, Any]]) -> None:
    sample_ids = [row["sample_id"] for row in rows]
    duplicated = [sid for sid, count in Counter(sample_ids).items() if count > 1]

    if duplicated:
        raise ValueError(f"Duplicated sample_id found: {duplicated[:10]}")


def check_labels_valid(rows: list[dict[str, Any]], num_labels: int = 7) -> None:
    invalid = [row for row in rows if row["label"] < 0 or row["label"] >= num_labels]

    if invalid:
        raise ValueError(f"Invalid labels found, examples: {invalid[:3]}")


def check_context_contains_current(rows: list[dict[str, Any]], max_examples: int = 3) -> None:
    bad_cases = []

    for row in rows:
        text = row["text"]
        context_text = row["context_text"]

        if text and text not in context_text:
            bad_cases.append(row)

        if len(bad_cases) >= max_examples:
            break

    if bad_cases:
        examples = [(row["sample_id"], row["text"], row["context_text"]) for row in bad_cases]
        raise ValueError(f"Some context_text fields do not contain current text: {examples}")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    emotion_counter = Counter(row["emotion"] for row in rows)
    speaker_counter = Counter(row["speaker"] for row in rows)

    return {
        "num_samples": len(rows),
        "num_dialogues": len(set(row["dialogue_id"] for row in rows)),
        "emotion_distribution": dict(emotion_counter),
        "top_speakers": dict(speaker_counter.most_common(10)),
    }


def run_basic_checks(rows: list[dict[str, Any]], num_labels: int = 7) -> dict[str, Any]:
    check_sample_ids_unique(rows)
    check_labels_valid(rows, num_labels=num_labels)
    check_context_contains_current(rows)

    return summarize_rows(rows)
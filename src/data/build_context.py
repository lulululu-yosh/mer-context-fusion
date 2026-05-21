from __future__ import annotations

from typing import Any


def normalize_text(text: Any) -> str:
    """Convert NaN / None into a string"""
    if text is None:
        return ""
    
    text_clean=str(text)

    if text.lower() == "nan":
        return ""
    
    return " ".join(text_clean.strip().split())


def build_context_text(dialogue_rows: list[dict[str, Any]], current_index: int, window_size: int = 3, include_current: bool = True, speaker_prefix: bool = True, sep_token: str = " [SEP] ") -> str:
    """
    Build context text for one utterance.
    Args:
        dialogue_rows:
            All utterances in one dialogue, already sorted by utterance_id.
        current_index:
            Index of current utterance in the dialogue.
        window_size:
            Number of previous utterances to include.
        include_current:
            Whether to include the current utterance itself.
        speaker_prefix:
            Whether to prepend speaker name.
        sep_token:
            Separator between utterances.
    """
    end = current_index + 1 if include_current else current_index
    start = max(0, current_index - window_size)

    selected_rows = dialogue_rows[start:end]

    parts = []
    for row in selected_rows:
        speaker = normalize_text(row.get("speaker", "Speaker")) or "Speaker"
        text = normalize_text(row.get("text", ""))

        if not text:
            continue

        if speaker_prefix:
            parts.append(f"{speaker}: {text}")
        else:
            parts.append(text)

    return sep_token.join(parts)

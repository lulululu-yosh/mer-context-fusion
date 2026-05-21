from __future__ import annotations


def build_context_text(dialogue_rows: list[dict], index: int, window: int = 3, speaker_prefix: bool = True) -> str:
    """Build previous-k utterance context plus current utterance."""
    start = max(0, index - window)
    selected = dialogue_rows[start : index + 1]

    parts = []
    for row in selected:
        text = str(row.get("text", "")).strip()
        speaker = str(row.get("speaker", "Speaker")).strip()
        if speaker_prefix:
            parts.append(f"{speaker}: {text}")
        else:
            parts.append(text)

    return " [SEP] ".join(parts)

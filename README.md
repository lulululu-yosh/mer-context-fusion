# Context-Aware Multimodal Emotion Recognition on MELD

This repository is a reproducible research-style MER project for comparing unimodal baselines, dialogue-context modeling, and multimodal fusion strategies on MELD.

## Project Goal

Build a context-aware multimodal emotion recognition system using:

- Text: RoBERTa / DeBERTa
- Audio: wav2vec2 / HuBERT / openSMILE
- Vision: ResNet / ViT / MediaPipe / OpenFace
- Fusion: early fusion, late fusion, attention fusion, reliability-aware fusion

The core research question is:

> How much does each modality contribute to emotion recognition, and can adaptive fusion improve robustness when one modality is noisy or weak?

## Task Definition

Input:

- Current utterance text
- Optional previous dialogue context
- Optional audio clip
- Optional video clip or visual features

Output:

- One of 7 MELD emotion labels:
  - anger
  - disgust
  - fear
  - joy
  - neutral
  - sadness
  - surprise

## Repository Structure

```text
mer-context-fusion/
├── configs/          # YAML configs for data, models, and experiments
├── data/             # Raw data, processed data, and feature cache
├── src/              # Reusable Python modules
├── scripts/          # CLI entry scripts
├── notebooks/        # Data analysis and error analysis notebooks
├── experiments/      # Per-experiment logs, metrics, configs
├── outputs/          # Figures, result tables, predictions
├── checkpoints/      # Local checkpoints; not tracked by Git
├── reports/          # Weekly notes and project summary
└── demo/             # Gradio / Streamlit demo
```

## Current Stage

Week 1 text baselines are implemented and can train/evaluate from the processed MELD JSONL files.
Week 2 adds frozen-feature unimodal audio and visual baselines for reproducible comparison.

Implemented baselines:

1. Text-only RoBERTa utterance baseline.
2. Text-only RoBERTa context baseline.
3. Audio-only wav2vec2/HuBERT cached-feature baseline.
4. Visual-only ResNet frame cached-feature baseline.

## How to Set Up

```bash
git clone <your-repo-url>
cd mer-context-fusion

python -m venv .venv
source .venv/bin/activate   # Windows Git Bash
# .venv\Scripts\activate    # Windows PowerShell

pip install -r requirements.txt
pip install -e .
```

## Expected Workflow

```bash
python scripts/prepare_meld.py --config configs/data/meld.yaml
python scripts/analyze_dataset.py --config configs/data/meld.yaml
python scripts/train_text.py --config configs/text/roberta_utt.yaml
python scripts/train_text.py --config configs/text/roberta_context_k3.yaml
python scripts/evaluate_checkpoint.py --config configs/text/roberta_context_k3.yaml
```

## Week 2 Audio and Visual Baselines

Place raw MELD metadata under `data/raw/MELD/` before running preprocessing. The
processed JSONL files should live at:

- `data/processed/train.jsonl`
- `data/processed/dev.jsonl`
- `data/processed/test.jsonl`

Audio/video files are intentionally not committed. By default the Week-2 configs look for:

- audio: `data/raw/MELD/audio/{split}/dia{dialogue_id}_utt{utterance_id}.wav`
- video: `data/raw/MELD/videos/{split}/dia{dialogue_id}_utt{utterance_id}.mp4`

If your local MELD media layout differs, edit the `paths.audio_pattern` and
`paths.video_pattern` values in the YAML config. If `audio_path` or `video_path` fields are
added directly to processed JSONL samples, those explicit paths are used first.

```bash
# Extract audio features
python scripts/extract_audio_features.py --config configs/audio/wav2vec2.yaml

# Train audio-only baseline
python scripts/train_audio.py --config configs/audio/wav2vec2.yaml

# Extract visual features
python scripts/extract_visual_features.py --config configs/visual/resnet_frames.yaml

# Train visual-only baseline
python scripts/train_visual.py --config configs/visual/resnet_frames.yaml

# Evaluate a cached-feature baseline on test
python scripts/evaluate_unimodal.py --config configs/audio/wav2vec2.yaml --split test

# Collect result table
python scripts/collect_results.py --experiments_dir experiments --output outputs/tables/unimodal_results.csv

# Plot a confusion matrix
python scripts/plot_confusion_matrix.py --input experiments/exp003_audio_wav2vec2

# Check local media availability
python scripts/analyze_missing_media.py --config configs/audio/wav2vec2.yaml
```

Feature caches are saved under `data/features/` and are ignored by Git. Each cache contains
`sample_id`, utterance metadata, label, embedding, and a modality missing flag. Missing or
corrupt media is logged beside the cache as `*.missing.tsv`; the default policy writes a zero
embedding so long runs do not crash.

Each experiment directory saves:

- `config.yaml`
- `metrics.json`
- `classification_report.txt`
- `predictions.csv`
- `confusion_matrix.csv`
- `metrics_history.csv`
- `train.log`
- `best_model.pt` (ignored by Git)

The following local artifacts should not be committed: `data/raw/`, `data/features/`,
`checkpoints/`, `*.pt`, `*.pth`, `*.ckpt`, media files, `wandb/`, and `runs/`.

## Metrics

Main metrics:

- Accuracy
- Macro-F1
- Weighted-F1
- Per-class F1
- Confusion matrix

Macro-F1 is emphasized because MELD is class-imbalanced.

## Planned Experiments

| Exp ID | Model | Modality | Context | Fusion |
|---|---|---|---|---|
| exp001 | RoBERTa | Text | no | none |
| exp002 | RoBERTa | Text | k=3 | none |
| exp003 | wav2vec2 / HuBERT | Audio | no | none |
| exp004 | ViT / ResNet | Vision | no | none |
| exp005 | RoBERTa + audio + vision | T+A+V | no | early concat |
| exp006 | RoBERTa + audio + vision | T+A+V | no | late fusion |
| exp007 | RoBERTa + audio + vision | T+A+V | no | attention fusion |
| exp008 | RoBERTa + audio + vision | T+A+V | no | reliability-aware fusion |

## Notes

Do not commit raw MELD data, feature tensors, checkpoints, or experiment cache files.

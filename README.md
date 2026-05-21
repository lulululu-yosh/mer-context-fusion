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

Week 1 focuses on:

1. Preparing MELD metadata.
2. Building utterance-level and context-aware samples.
3. Training a text-only RoBERTa baseline.
4. Evaluating accuracy, macro-F1, weighted-F1, and confusion matrix.

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

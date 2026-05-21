# Project Summary

## Problem

Multimodal emotion recognition aims to classify emotion from text, audio, visual, and dialogue-context signals.

## Dataset

MELD, 7-class utterance-level emotion classification.

## Method

Planned models:

1. Text-only RoBERTa baseline
2. Context-aware RoBERTa baseline
3. Audio-only wav2vec2 / HuBERT baseline
4. Visual-only ViT / ResNet baseline
5. Early fusion
6. Late fusion
7. Attention fusion
8. Reliability-aware fusion

## Evaluation

- Accuracy
- Macro-F1
- Weighted-F1
- Per-class F1
- Confusion matrix
- Missing-modality robustness

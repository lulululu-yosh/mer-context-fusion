#!/usr/bin/env bash
set -e

mkdir -p \
  configs/data configs/text configs/audio configs/visual configs/fusion configs/experiments \
  data/raw data/interim data/processed data/features/text data/features/audio data/features/visual data/features/multimodal \
  src/data src/features src/encoders src/models src/losses src/training src/evaluation src/visualization src/utils \
  scripts notebooks experiments outputs/figures outputs/tables outputs/predictions checkpoints reports/weekly_notes demo/assets

touch outputs/figures/.gitkeep outputs/tables/.gitkeep outputs/predictions/.gitkeep
echo "Directory structure initialized."

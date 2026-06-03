# Week 2 Audio and Visual Baseline Audit

## Files Added or Changed

- Added audio configs: `configs/audio/wav2vec2.yaml`, `configs/audio/hubert.yaml`.
- Added visual config: `configs/visual/resnet_frames.yaml`.
- Added cached-feature dataset: `src/data/feature_dataset.py`.
- Added reusable MLP classifier: `src/models/unimodal_mlp.py`.
- Added feature extraction utilities under `src/features/`.
- Added training/evaluation/analysis scripts under `scripts/`.
- Updated `src/training/trainer.py` to support cached feature batches and save paper-ready artifacts.
- Updated `src/evaluation/metrics.py` to include per-class metrics and confusion matrices.
- Updated `README.md` with Week-2 commands, media layout, artifact policy, and expected outputs.

## Trainer Reuse

Trainer reuse was successful. The existing trainer now accepts either text batches
(`input_ids`, `attention_mask`) or cached-feature batches (`features`) while keeping the same
optimizer, scheduler, best-checkpoint selection, and experiment output style.

## Audio Baseline Readiness

The audio baseline is implemented around frozen Hugging Face speech encoders such as
`facebook/wav2vec2-base` and `facebook/hubert-base-ls960`. It mean-pools valid hidden states,
caches utterance-level embeddings, logs missing or corrupt files, and trains an MLP classifier
from cached `.pt` files.

## Visual Baseline Readiness

The visual baseline is implemented around frozen torchvision ResNet encoders. It samples a fixed
number of frames per utterance video, mean-pools frame embeddings, caches utterance-level features,
logs missing or corrupt videos, and trains an MLP classifier from cached `.pt` files.

## Paper Output Files

Each experiment saves:

- `config.yaml`
- `metrics.json`
- `classification_report.txt`
- `predictions.csv`
- `confusion_matrix.csv`
- `metrics_history.csv`
- `train.log`
- `best_model.pt` (ignored by Git)

Analysis utilities also save:

- `outputs/tables/unimodal_results.csv`
- `outputs/tables/media_availability.csv`
- `outputs/figures/confusion_matrix_<exp_name>.png`

## Known Limitations

- Feature extraction can be slow and may require model downloads.
- The processed JSONL currently does not include explicit `audio_path` or `video_path` fields, so
  configs provide MELD-style path patterns. Users with a different media layout should edit those
  patterns.
- Missing media defaults to zero embeddings with modality missing flags. This keeps runs robust but
  should be reported in paper tables.
- The local shell was not configured with an editable install, so direct `python scripts/...` imports
  required `PYTHONPATH=.` during smoke verification. The README setup still recommends
  `pip install -e .`.

## Commands Verified

- `python -m compileall src scripts`
- Tiny cached-feature smoke training with synthetic `.pt` files for the audio training path.

Full audio/visual extraction was not run because it requires local MELD media and external encoder
downloads.

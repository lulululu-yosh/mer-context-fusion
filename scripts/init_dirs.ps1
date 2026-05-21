$dirs = @(
  "configs/data", "configs/text", "configs/audio", "configs/visual", "configs/fusion", "configs/experiments",
  "data/raw", "data/interim", "data/processed", "data/features/text", "data/features/audio", "data/features/visual", "data/features/multimodal",
  "src/data", "src/features", "src/encoders", "src/models", "src/losses", "src/training", "src/evaluation", "src/visualization", "src/utils",
  "scripts", "notebooks", "experiments", "outputs/figures", "outputs/tables", "outputs/predictions", "checkpoints", "reports/weekly_notes", "demo/assets"
)

foreach ($dir in $dirs) {
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

New-Item -ItemType File -Force -Path "outputs/figures/.gitkeep" | Out-Null
New-Item -ItemType File -Force -Path "outputs/tables/.gitkeep" | Out-Null
New-Item -ItemType File -Force -Path "outputs/predictions/.gitkeep" | Out-Null

Write-Host "Directory structure initialized."

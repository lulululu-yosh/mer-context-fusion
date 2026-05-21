# Data Directory

Place MELD files under:

```text
data/raw/MELD/
```

Expected CSV files:

```text
train_sent_emo.csv
dev_sent_emo.csv
test_sent_emo.csv
```

Generated files:

```text
data/processed/train.jsonl
data/processed/dev.jsonl
data/processed/test.jsonl
```

Do not commit raw data, processed feature tensors, or model checkpoints.

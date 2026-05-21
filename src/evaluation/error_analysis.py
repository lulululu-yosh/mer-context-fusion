from __future__ import annotations

import pandas as pd


def collect_wrong_cases(predictions_csv: str, output_csv: str) -> None:
    df = pd.read_csv(predictions_csv)
    wrong = df[df["gold"] != df["pred"]]
    wrong.to_csv(output_csv, index=False)

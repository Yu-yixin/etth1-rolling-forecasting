from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    processed_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "ETTh1_scaled.parquet"
    frame = pd.read_parquet(processed_path)
    print(frame.head())
    print(frame.tail())
    print(frame.describe().transpose()[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()

"""Download the UCI Heart Disease dataset and save a clean raw CSV."""
from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

BASE = Path(__file__).parent
OUT = BASE / "data" / "heart_disease.csv"

def main():
    print("Downloading UCI Heart Disease dataset...")
    dataset = fetch_ucirepo(id=45)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    df = pd.concat([X, y], axis=1)
    # UCI target column is normally named 'num'.
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "num" not in df.columns:
        # Support a single target column with a different name.
        target_cols = [c for c in df.columns if c not in X.columns.str.lower()]
        if len(target_cols) == 1:
            df = df.rename(columns={target_cols[0]: "num"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows and {len(df.columns)-1} features to: {OUT}")
    print(df.head())
    print("\nMissing values:\n", df.isna().sum())
    print("\nTarget distribution:\n", df["num"].value_counts(dropna=False).sort_index())
    print("\nUCI source: https://doi.org/10.24432/C52P4X")

if __name__ == "__main__":
    main()

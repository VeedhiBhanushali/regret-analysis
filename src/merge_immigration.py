import pandas as pd
from pathlib import Path

CLEAN_DIR = Path("data_clean")

files = list(CLEAN_DIR.glob("*_regret_*.csv"))
files = [f for f in files if any(x in f.name for x in ["immigration", "IWantOut", "USCIS", "visa"])]

dfs = []
for f in files:
    df = pd.read_csv(f)
    df["domain"] = "immigration"
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["id"])
merged.to_csv(CLEAN_DIR / "immigration_all_regret.csv", index=False)

print("Total immigration regret rows:", len(merged))
import pandas as pd
from pathlib import Path

CLEAN_DIR = Path("data_clean")

files = list(CLEAN_DIR.glob("*_regret_*.csv"))

dfs = []
for f in files:
    df = pd.read_csv(f)
    df["source_subreddit"] = f.name.split("_")[0]
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["id"])

merged.to_csv(CLEAN_DIR / "career_all_regret.csv", index=False)

print("Total merged rows:", len(merged))

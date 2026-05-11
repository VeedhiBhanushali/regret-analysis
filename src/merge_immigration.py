import pandas as pd
from pathlib import Path

CLEAN_DIR = Path("data_clean")

files = list(CLEAN_DIR.glob("*_regret_*.csv"))
IMMIGRATION_KEYWORDS = ["immigration", "IWantOut", "USCIS", "visa", "greencard", "h1b", "f1visa"]
files = [f for f in files if any(x.lower() in f.name.lower() for x in IMMIGRATION_KEYWORDS)]

dfs = []
for f in files:
    df = pd.read_csv(f)
    df["domain"] = "immigration"
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["id"])
merged.to_csv(CLEAN_DIR / "immigration_all_regret.csv", index=False)

print("Total immigration regret rows:", len(merged))
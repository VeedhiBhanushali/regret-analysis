import pandas as pd
from pathlib import Path

CLEAN_DIR = Path("data_clean")

# Grab relationship regret files you just created
files = list(CLEAN_DIR.glob("*_stable_regret_*.csv"))
files = [f for f in files if any(x in f.name.lower() for x in ["relationships", "relationship_advice"])]

dfs = []
for f in files:
    df = pd.read_csv(f)
    df["domain"] = "relationships"
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["id"])
out = CLEAN_DIR / "relationships_all_regret.csv"
merged.to_csv(out, index=False)

print("Saved:", out)
print("Total relationship regret rows:", len(merged))
print("Unique IDs:", merged["id"].nunique())
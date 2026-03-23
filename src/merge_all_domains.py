import pandas as pd
from pathlib import Path

CLEAN_DIR = Path("data_clean")

paths = {
    "career": CLEAN_DIR / "career_all_regret.csv",
    "immigration": CLEAN_DIR / "immigration_all_regret.csv",
    "relationships": CLEAN_DIR / "relationships_all_regret.csv",
}

dfs = []
for domain, path in paths.items():
    df = pd.read_csv(path)
    df["domain"] = domain
    dfs.append(df)

master = pd.concat(dfs, ignore_index=True)

# In case a post somehow appears in 2 domains, keep first occurrence
master = master.drop_duplicates(subset=["id"])

out = CLEAN_DIR / "all_domains_regret.csv"
master.to_csv(out, index=False)

print("Saved:", out)
print("Total rows:", len(master))
print("Unique IDs:", master["id"].nunique())
print("\nCounts by domain:\n", master["domain"].value_counts())
print("\nCounts by subreddit:\n", master["source_subreddit"].value_counts().head(20))
import re
import pandas as pd
from pathlib import Path

import sys

if len(sys.argv) < 2:
    print("Usage: python filter_regret_posts_stable.py subreddit_name")
    sys.exit(1)

SUBREDDIT = sys.argv[1]


RAW_DIR = Path("data_raw")
CLEAN_DIR = Path("data_clean")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

KEEP_PATTERNS = [
    r"\bi regret\b",
    r"\bi regret it\b",
    r"\bi regret that\b",
    r"\bi wish i had\b",
    r"\bi messed up\b",
    r"\bi fucked up\b",
    r"\bmy biggest mistake\b",
]

EXCLUDE_PATTERNS = [
    r"\bwill i regret\b",
    r"\bwhat if i regret\b",
    r"\bbefore i regret\b",
    r"\bmight regret\b",
    r"\byou('ll| will) regret\b",
    r"\bthey('ll| will) regret\b",
]

def hit_any(patterns, text):
    for pat in patterns:
        if re.search(pat, text):
            return True
    return False

def main():
    raw_files = sorted(list(RAW_DIR.glob(f"{SUBREDDIT}_buckets_*.csv")))
    df = pd.read_csv(raw_files[-1])

    df["full_text"] = (df["title"].fillna("") + "\n" + df["selftext"].fillna("")).astype(str).str.lower()

    df["keep"] = df["full_text"].apply(lambda t: hit_any(KEEP_PATTERNS, t))
    df["exclude"] = df["full_text"].apply(lambda t: hit_any(EXCLUDE_PATTERNS, t))

    df_final = df[(df["keep"]) & (~df["exclude"])].copy()

    out_path = CLEAN_DIR / f"{SUBREDDIT}_stable_regret_{len(df_final)}.csv"
    df_final.to_csv(out_path, index=False)

    print(f"Raw rows: {len(df)}")
    print(f"Regret rows: {len(df_final)}")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    main()

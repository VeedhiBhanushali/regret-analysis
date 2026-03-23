import pandas as pd
from pathlib import Path

CLEAN_DIR = Path("data_clean")

FILES = [
    ("career", CLEAN_DIR / "career_all_structured.csv"),
    ("immigration", CLEAN_DIR / "immigration_all_structured.csv"),
    ("relationships", CLEAN_DIR / "relationships_all_structured.csv"),
]

OUT_PATH = CLEAN_DIR / "all_domains_structured_master.csv"

def main():
    dfs = []
    for domain, path in FILES:
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")

        df = pd.read_csv(path)

        # Ensure domain column exists and is correct
        df["domain"] = domain

        dfs.append(df)

    master = pd.concat(dfs, ignore_index=True)

    # If a post somehow appears in multiple domains, keep first
    master = master.drop_duplicates(subset=["id"], keep="first")

    # Make schema consistent across domains:
    # If some columns exist in one domain but not others, fill missing with NaN.
    # (pandas already does this on concat, but we’ll also enforce a clean column order)

    preferred_order = [
        "id", "domain", "source_subreddit", "subreddit",
        "title", "selftext", "full_text",
        "created_utc", "created_dt", "created_dt_parsed", "post_year",
        "num_comments", "score", "permalink", "url",
        "keep", "exclude",
        "regret_sentence", "reversal", "urgency_score",
        "time_to_regret_days",
        "topic", "negated_regret"
    ]

    # Add any extra columns at the end (so nothing gets dropped)
    cols = list(master.columns)
    final_cols = [c for c in preferred_order if c in cols] + [c for c in cols if c not in preferred_order]
    master = master[final_cols]

    master.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Total rows:", len(master))
    print("Unique IDs:", master["id"].nunique())
    print("\nBy domain:\n", master["domain"].value_counts())

    # Quick sanity checks
    if "time_to_regret_days" in master.columns:
        print("\nTime-known % by domain:\n", master.groupby("domain")["time_to_regret_days"].apply(lambda x: x.notnull().mean()))
    if "reversal" in master.columns:
        print("\nReversal rate by domain:\n", master.groupby("domain")["reversal"].mean())

if __name__ == "__main__":
    main()
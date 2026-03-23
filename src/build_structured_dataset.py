import re
import pandas as pd
from pathlib import Path

SUBREDDIT = "careeradvice"

INPUT_PATH = Path("data_clean/career_all_regret.csv")
OUTPUT_PATH = Path("data_clean/career_all_dataset.csv")

# Reversal indicators
REVERSAL_WORDS = [
    "quit",
    "left",
    "resigned",
    "divorced",
    "moved back",
    "withdrew",
    "dropped out",
    "changed careers",
    "changed jobs",
    "switched",
]

# Urgency indicators
URGENCY_WORDS = [
    "rushed",
    "pressure",
    "deadline",
    "last minute",
    "forced",
    "urgent",
    "quick decision",
    "panicked",
    "impulsive",
]

def count_matches(words, text):
    count = 0
    for w in words:
        if w in text:
            count += 1
    return count

def extract_time_to_regret(text, post_year):
    # Pattern 1: "6 months later"
    match = re.search(r"(\d+)\s+(day|week|month|year)s?\s+later", text)
    if match:
        number = int(match.group(1))
        unit = match.group(2)
        if unit == "day":
            return number
        if unit == "week":
            return number * 7
        if unit == "month":
            return number * 30
        if unit == "year":
            return number * 365

    # Pattern 2: "after 6 months"
    match = re.search(r"after\s+(\d+)\s+(day|week|month|year)s?", text)
    if match:
        number = int(match.group(1))
        unit = match.group(2)
        if unit == "day":
            return number
        if unit == "week":
            return number * 7
        if unit == "month":
            return number * 30
        if unit == "year":
            return number * 365

    # Pattern 3: "in 2021", "back in 2019"
    match = re.search(r"(in|back in)\s+(20\d{2})", text)
    if match:
        decision_year = int(match.group(2))
        if decision_year < post_year:
            return (post_year - decision_year) * 365

    return None

def main():
    df = pd.read_csv(INPUT_PATH)

    df["full_text"] = (df["title"].fillna("") + "\n" + df["selftext"].fillna("")).astype(str).str.lower()

    # Reversal indicator
    df["reversal"] = df["regret_sentence"].fillna("").apply(
    lambda t: 1 if count_matches(REVERSAL_WORDS, t) > 0 else 0
)

    # Urgency score
    df["urgency_score"] = df["full_text"].apply(lambda t: count_matches(URGENCY_WORDS, t))

    df["post_year"] = pd.to_datetime(df["created_dt"]).dt.year
    # Time to regret
    df["time_to_regret_days"] = df.apply(
    lambda row: extract_time_to_regret(row["full_text"], row["post_year"]),
    axis=1)

    # Decision phrase (basic extraction: sentence with regret)
    df["decision_sentence"] = df["full_text"].apply(
        lambda t: next((s.strip() for s in t.split(".") if "regret" in s), "")
    )

    df.to_csv(OUTPUT_PATH, index=False)

    print("Structured dataset saved.")
    print("Total rows:", len(df))
    print("Reversal rate:", df["reversal"].mean())
    print("Avg urgency score:", df["urgency_score"].mean())
    print("Time-to-regret non-null:", df["time_to_regret_days"].notnull().sum())

if __name__ == "__main__":
    main()

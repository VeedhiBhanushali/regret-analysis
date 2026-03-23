import re
import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data_clean/relationships_all_regret.csv")
OUTPUT_PATH = Path("data_clean/relationships_all_structured.csv")
# Actions that indicate a decision/reversal event in career context
REVERSAL_ACTIONS = [
    "quit", "left", "resigned", "declined", "turned down", "rejected",
    "fired", "laid off", "layoff", "terminated",
    "withdrew", "dropped out", "moved back", "switched",
    "changed jobs", "changed careers"
]

URGENCY_WORDS = [
    "rushed", "pressure", "deadline", "last minute", "forced", "urgent",
    "quick decision", "panicked", "impulsive", "no time", "asap"
]

# If the post explicitly says they don't regret, we should not treat it as regret-linked reversal.
NOT_REGRET_PHRASES = [
    "don't regret",
    "do not regret",
    "no regret",
    "not regret",
    "don't really regret",
    "do not really regret",
]

KEEP_TIME_MAX_DAYS = 3650  # cap at 10 years

def contains_any(text: str, phrases) -> bool:
    return any(p in text for p in phrases)

def count_matches(words, text):
    return sum(1 for w in words if w in text)

def to_days(n, unit):
    unit = unit.lower()
    if unit.startswith("day"):
        return n
    if unit.startswith("week"):
        return n * 7
    if unit.startswith("month"):
        return n * 30
    if unit.startswith("year"):
        return n * 365
    return None

def extract_time_to_regret(text, post_year):
    # 1) "6 months later"
    m = re.search(r"(\d+)\s+(day|week|month|year)s?\s+later", text)
    if m:
        return to_days(int(m.group(1)), m.group(2))

    # 2) "after 6 months"
    m = re.search(r"after\s+(\d+)\s+(day|week|month|year)s?", text)
    if m:
        return to_days(int(m.group(1)), m.group(2))

    # 3) "it's been 2 years"
    m = re.search(r"(it'?s been|has been)\s+(\d+)\s+(day|week|month|year)s?", text)
    if m:
        return to_days(int(m.group(2)), m.group(3))

    # 4) year mention: "in 2021", "back in 2019"
    m = re.search(r"(in|back in)\s+(20\d{2})", text)
    if m:
        decision_year = int(m.group(2))
        if decision_year < post_year:
            return (post_year - decision_year) * 365

    return None

def extract_regret_sentence(full_text):
    # pick first sentence containing regret-ish words
    parts = re.split(r"[.\n!?]", full_text)
    for s in parts:
        if "regret" in s or "wish i had" in s or "messed up" in s or "fucked up" in s:
            s = s.strip()
            if len(s) >= 30:
                return s[:500]
    return ""

def main():
    df = pd.read_csv(INPUT_PATH)

    df["title"] = df["title"].fillna("").astype(str)
    df["selftext"] = df["selftext"].fillna("").astype(str)
    df["full_text"] = (df["title"] + "\n" + df["selftext"]).str.lower()

    # created_dt may exist; if not, compute from created_utc
    if "created_dt" in df.columns and df["created_dt"].notnull().any():
        df["created_dt_parsed"] = pd.to_datetime(df["created_dt"], errors="coerce", utc=True)
    else:
        df["created_dt_parsed"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce", utc=True)

    df["post_year"] = df["created_dt_parsed"].dt.year.fillna(2024).astype(int)

    # Extract a regret-centric sentence first
    df["regret_sentence"] = df["full_text"].apply(extract_regret_sentence)

    # Mark posts that explicitly negate regret
    df["negated_regret"] = df["full_text"].apply(lambda t: contains_any(t, NOT_REGRET_PHRASES))

    # ✅ Updated reversal logic:
    # reversal = 1 only if:
    # - there is an action word somewhere in the post (quit/left/declined/etc.)
    # - there is a regret_sentence (so it’s actually a regret-type post in our extraction)
    # - it does NOT explicitly say "don't regret" etc.
    df["reversal"] = df.apply(
        lambda r: 1
        if (contains_any(r["full_text"], REVERSAL_ACTIONS)
            and len(str(r["regret_sentence"])) > 0
            and not bool(r["negated_regret"]))
        else 0,
        axis=1
    )

    df["urgency_score"] = df["full_text"].apply(lambda t: count_matches(URGENCY_WORDS, t))

    df["time_to_regret_days"] = df.apply(
        lambda r: extract_time_to_regret(r["full_text"], r["post_year"]),
        axis=1
    )

    # cap extreme times
    df.loc[df["time_to_regret_days"].notnull() & (df["time_to_regret_days"] > KEEP_TIME_MAX_DAYS), "time_to_regret_days"] = None

    # Drop helper column (optional)
    # keep it if you want to report how many got excluded due to "don't regret"
    # df = df.drop(columns=["negated_regret"])

    df.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("Rows:", len(df))
    print("Time non-null:", df["time_to_regret_days"].notnull().sum())
    print("Reversal rate:", df["reversal"].mean())
    print("Negated regret count:", df["negated_regret"].sum())

if __name__ == "__main__":
    main()

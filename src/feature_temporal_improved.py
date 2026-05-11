"""
Improve time-to-regret extraction using dateparser and additional regex
patterns beyond the original 4-pattern approach.
Updates data_clean/all_domains_enriched.csv in place.
"""
import re
import pandas as pd
from pathlib import Path
from datetime import datetime

INPUT_PATH = Path("data_clean/all_domains_enriched.csv")


UNIT_TO_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}

EXTRA_PATTERNS = [
    # "a year later", "a month ago"
    (r"\ba\s+(day|week|month|year)\s+(later|ago)", lambda m: UNIT_TO_DAYS[m.group(1)]),
    # "a couple of years", "a couple months"
    (r"a couple(?:\s+of)?\s+(day|week|month|year)s?", lambda m: 2 * UNIT_TO_DAYS[m.group(1)]),
    # "a few months later"
    (r"a few\s+(day|week|month|year)s?", lambda m: 3 * UNIT_TO_DAYS[m.group(1)]),
    # "several years"
    (r"several\s+(day|week|month|year)s?", lambda m: 5 * UNIT_TO_DAYS[m.group(1)]),
    # "half a year"
    (r"half a\s+(year|month)", lambda m: UNIT_TO_DAYS[m.group(1)] // 2),
    # "X years ago", "X months ago"
    (r"(\d+)\s+(day|week|month|year)s?\s+ago", lambda m: int(m.group(1)) * UNIT_TO_DAYS[m.group(2)]),
    # "over X years"
    (r"over\s+(\d+)\s+(day|week|month|year)s?", lambda m: int(m.group(1)) * UNIT_TO_DAYS[m.group(2)]),
    # "almost X years"
    (r"almost\s+(\d+)\s+(day|week|month|year)s?", lambda m: int(m.group(1)) * UNIT_TO_DAYS[m.group(2)]),
    # "last year", "last month"
    (r"\blast\s+(year|month|week)", lambda m: UNIT_TO_DAYS[m.group(1)]),
]

MONTH_YEAR_PATTERN = re.compile(
    r"(?:in|since|around|back in)\s+"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+(20\d{2})",
    re.IGNORECASE,
)


def extract_time_improved(text, post_year):
    """Try additional patterns beyond the original regex set."""
    if not isinstance(text, str) or len(text) < 10:
        return None

    for pattern_str, compute_fn in EXTRA_PATTERNS:
        m = re.search(pattern_str, text, re.IGNORECASE)
        if m:
            days = compute_fn(m)
            if days and 1 <= days <= 3650:
                return days

    m = MONTH_YEAR_PATTERN.search(text)
    if m:
        year_mentioned = int(m.group(1))
        if year_mentioned < post_year:
            return (post_year - year_mentioned) * 365

    return None


def main():
    df = pd.read_csv(INPUT_PATH)

    already_has_time = df["time_to_regret_days"].notnull()
    before_count = already_has_time.sum()

    needs_time = ~already_has_time
    new_times = df.loc[needs_time].apply(
        lambda row: extract_time_improved(
            row["full_text"] if pd.notnull(row.get("full_text")) else "",
            int(row["post_year"]) if pd.notnull(row.get("post_year")) else 2024,
        ),
        axis=1,
    )
    df.loc[needs_time, "time_to_regret_days"] = new_times

    after_count = df["time_to_regret_days"].notnull().sum()
    new_extractions = after_count - before_count

    df.to_csv(INPUT_PATH, index=False)

    total = len(df)
    print(f"Before: {before_count}/{total} ({before_count/total:.1%}) had time info")
    print(f"New extractions: {new_extractions}")
    print(f"After:  {after_count}/{total} ({after_count/total:.1%}) have time info")
    print(f"Updated {INPUT_PATH}")


if __name__ == "__main__":
    main()

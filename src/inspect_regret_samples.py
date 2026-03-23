import re
import random
import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data_clean/careeradvice_stable_regret_399.csv")
N = 15  # how many examples to print

STRONG_REGRET = [
    r"\bi regret\b",
    r"\bi wish i had\b",
    r"\bi wish i'd\b",
    r"\bif only i had\b",
    r"\bbiggest mistake\b",
    r"\bshouldn't have\b",
    r"\bshould not have\b",
    r"\bi messed up\b",
    r"\bi fucked up\b",
]

PAST_ACTION_VERBS = [
    "took","accepted","chose","married","moved","quit","left","applied","filed",
    "decided","stayed","joined","signed","changed","started","ended","broke","broke up",
    "got","got fired","turned down"
]

def find_first_match(text, patterns):
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return pat, m.start(), m.end()
    return None, None, None

def find_action_before(text, regret_start):
    before = text[:regret_start]
    # simple “contains” search; we’ll improve later
    for v in PAST_ACTION_VERBS:
        if v in before:
            return v
    return None

def snippet(text, start, window=90):
    a = max(0, start - window)
    b = min(len(text), start + window)
    return text[a:b].replace("\n", " ")

def main():
    df = pd.read_csv(INPUT_PATH)
    df["full_text"] = (df["title"].fillna("") + "\n" + df["selftext"].fillna("")).astype(str).str.lower()

    idxs = list(df.index)
    random.shuffle(idxs)
    idxs = idxs[:min(N, len(idxs))]

    for i in idxs:
        row = df.loc[i]
        text = row["full_text"]

        pat, s, e = find_first_match(text, STRONG_REGRET)
        action = find_action_before(text, s) if s is not None else None

        print("\n" + "="*90)
        print("TITLE:", row.get("title", "")[:160])
        print("MATCHED_REGRET_PATTERN:", pat)
        print("MATCHED_ACTION_BEFORE:", action)
        if s is not None:
            print("SNIPPET:", snippet(text, s))
        else:
            print("SNIPPET: <no regret match?>")

if __name__ == "__main__":
    main()


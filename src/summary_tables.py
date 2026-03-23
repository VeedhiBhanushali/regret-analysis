import pandas as pd
from pathlib import Path

INP = Path("data_clean/career_all_structured.csv")
TOPICS = Path("data_clean/career_topics.csv")

def main():
    df = pd.read_csv(INP)
    topics = pd.read_csv(TOPICS)

    print("\n=== Dataset ===")
    print("Rows:", len(df))
    print("Reversal rate:", df["reversal"].mean())
    print("Time-known:", df["time_to_regret_days"].notnull().sum())

    print("\n=== Time to regret (days) ===")
    dft = df[df["time_to_regret_days"].notnull()].copy()
    print("Median:", dft["time_to_regret_days"].median())
    print("Mean:", dft["time_to_regret_days"].mean())
    print("Median reversal=1:", dft[dft["reversal"] == 1]["time_to_regret_days"].median())
    print("Median reversal=0:", dft[dft["reversal"] == 0]["time_to_regret_days"].median())

    if "topic" in df.columns:
        print("\n=== Topic frequencies ===")
        freq = df["topic"].value_counts().rename_axis("topic").reset_index(name="count")
        freq = freq.merge(topics, on="topic", how="left").sort_values("count", ascending=False)
        print(freq.head(10).to_string(index=False))

if __name__ == "__main__":
    main()

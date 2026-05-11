"""
Add VADER sentiment scores to the structured master CSV.
Produces data_clean/all_domains_enriched.csv.
"""
import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

INPUT_PATH = Path("data_clean/all_domains_structured_master.csv")
OUTPUT_PATH = Path("data_clean/all_domains_enriched.csv")


def main():
    df = pd.read_csv(INPUT_PATH)
    analyzer = SentimentIntensityAnalyzer()

    text_col = df["regret_sentence"].fillna("").astype(str)
    text_col = text_col.where(text_col.str.len() > 0, df["title"].fillna("").astype(str))

    scores = text_col.apply(analyzer.polarity_scores)
    df["vader_compound"] = scores.apply(lambda d: d["compound"])
    df["vader_neg"] = scores.apply(lambda d: d["neg"])
    df["vader_pos"] = scores.apply(lambda d: d["pos"])
    df["vader_neu"] = scores.apply(lambda d: d["neu"])

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(f"vader_compound mean: {df['vader_compound'].mean():.3f}")
    print(f"vader_neg mean:      {df['vader_neg'].mean():.3f}")
    print(f"Correlation vader_compound <-> reversal: {df['vader_compound'].corr(df['reversal']):.3f}")
    print(f"Correlation vader_neg <-> reversal:      {df['vader_neg'].corr(df['reversal']):.3f}")


if __name__ == "__main__":
    main()

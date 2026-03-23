import pandas as pd

df = pd.read_csv("data_clean/career_all_structured.csv")

print("Total rows:", len(df))
print("Reversal rate:", df["reversal"].mean())
print("Time-known %:", df["time_to_regret_days"].notnull().mean())

print("\nBy Subreddit:")
grouped = df.groupby("source_subreddit").agg({
    "id": "count",
    "reversal": "mean",
    "time_to_regret_days": lambda x: x.notnull().mean()
}).rename(columns={
    "id": "count",
    "reversal": "reversal_rate",
    "time_to_regret_days": "time_known_rate"
})

print(grouped)

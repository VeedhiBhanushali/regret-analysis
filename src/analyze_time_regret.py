import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_PATH = Path("data_clean/career_all_dataset.csv")

def main():
    df = pd.read_csv(INPUT_PATH)

    time_df = df[df["time_to_regret_days"].notnull()].copy()
    time_df = time_df[time_df["time_to_regret_days"] <= 3650]


    print("Total regret cases:", len(df))
    print("Cases with time-to-regret:", len(time_df))

    if len(time_df) == 0:
        print("No time data available.")
        return

    print("Median days to regret:", time_df["time_to_regret_days"].median())
    print("Mean days to regret:", time_df["time_to_regret_days"].mean())

    reversal_df = time_df[time_df["reversal"] == 1]
    non_reversal_df = time_df[time_df["reversal"] == 0]

    print("Median time (reversal cases):", reversal_df["time_to_regret_days"].median())
    print("Median time (non-reversal cases):", non_reversal_df["time_to_regret_days"].median())

    from scipy.stats import mannwhitneyu

    stat, p = mannwhitneyu(
    reversal_df["time_to_regret_days"],
    non_reversal_df["time_to_regret_days"],
    alternative="two-sided"
)

    print("Mann-Whitney U p-value:", p)

    effect_size = (
    non_reversal_df["time_to_regret_days"].median()
    - reversal_df["time_to_regret_days"].median()
)

    print("Median difference (days):", effect_size)

    # Plot histogram
    plt.hist(time_df["time_to_regret_days"], bins=10)
    plt.title("Distribution of Time to Regret (Days)")
    plt.xlabel("Days")
    plt.ylabel("Frequency")
    plt.show()

if __name__ == "__main__":
    main()

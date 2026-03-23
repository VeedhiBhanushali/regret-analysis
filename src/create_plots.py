"""
Generate EDA plots for all_domains_structured_master.csv and save to plots/
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_PATH = Path("data_clean/all_domains_structured_master.csv")
PLOTS_DIR = Path("plots")


def main():
    PLOTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    time_df = df[df["time_to_regret_days"].notnull()]

    sns.set(style="whitegrid")

    # 1. Distribution of Urgency Score
    plt.figure()
    plt.hist(df["urgency_score"], bins=10)
    plt.title("Distribution of Urgency Score")
    plt.xlabel("Urgency Score")
    plt.ylabel("Frequency")
    plt.savefig(PLOTS_DIR / "1_urgency_score_hist.png", bbox_inches="tight")
    plt.close()

    # 2. Distribution of Time to Regret (Days)
    plt.figure()
    plt.hist(time_df["time_to_regret_days"], bins=15)
    plt.title("Distribution of Time to Regret (Days)")
    plt.xlabel("Days")
    plt.ylabel("Frequency")
    plt.savefig(PLOTS_DIR / "2_time_to_regret_hist.png", bbox_inches="tight")
    plt.close()

    # 3. Regret Posts by Domain
    plt.figure()
    df["domain"].value_counts().plot(kind="bar")
    plt.title("Regret Posts by Domain")
    plt.xlabel("Domain")
    plt.ylabel("Count")
    plt.savefig(PLOTS_DIR / "3_posts_by_domain.png", bbox_inches="tight")
    plt.close()

    # 4. Reversal Rate by Domain
    plt.figure()
    df.groupby("domain")["reversal"].mean().plot(kind="bar")
    plt.title("Reversal Rate by Domain")
    plt.xlabel("Domain")
    plt.ylabel("Reversal Rate")
    plt.savefig(PLOTS_DIR / "4_reversal_rate_by_domain.png", bbox_inches="tight")
    plt.close()

    # 5. Urgency Score vs Reversal
    plt.figure()
    plt.scatter(df["urgency_score"], df["reversal"])
    plt.title("Urgency Score vs Reversal")
    plt.xlabel("Urgency Score")
    plt.ylabel("Reversal (0/1)")
    plt.savefig(PLOTS_DIR / "5_urgency_vs_reversal.png", bbox_inches="tight")
    plt.close()

    # 6. Time to Regret by Domain (boxplot)
    plt.figure()
    sns.boxplot(x="domain", y="time_to_regret_days", data=time_df)
    plt.title("Time to Regret by Domain")
    plt.savefig(PLOTS_DIR / "6_time_to_regret_by_domain.png", bbox_inches="tight")
    plt.close()

    # 7. Correlation Heatmap
    numeric_cols = ["urgency_score", "reversal", "num_comments", "score"]
    corr = df[numeric_cols].corr()
    plt.figure()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(PLOTS_DIR / "7_correlation_heatmap.png", bbox_inches="tight")
    plt.close()

    # 8. Urgency Score by Domain
    plt.figure()
    sns.boxplot(x="domain", y="urgency_score", data=df)
    plt.title("Urgency Score by Domain")
    plt.savefig(PLOTS_DIR / "8_urgency_by_domain.png", bbox_inches="tight")
    plt.close()

    # 9. Time-to-Regret Availability by Domain
    time_known = df.groupby("domain")["time_to_regret_days"].apply(lambda x: x.notnull().mean())
    plt.figure()
    time_known.plot(kind="bar")
    plt.title("Time-to-Regret Availability by Domain")
    plt.ylabel("Percentage with Time Info")
    plt.savefig(PLOTS_DIR / "9_time_availability_by_domain.png", bbox_inches="tight")
    plt.close()

    print(f"Saved 9 plots to {PLOTS_DIR.absolute()}")


if __name__ == "__main__":
    main()

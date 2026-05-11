"""
Generate new analysis plots from the enriched dataset.
Plots 10-11 (VADER), 16 (KM by domain).
Plots 12 (topic heatmap), 13-14 (PR/calibration), 15 (feature importance)
are produced by their respective scripts.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from lifelines import KaplanMeierFitter

INPUT_PATH = Path("data_clean/all_domains_enriched.csv")
PLOTS_DIR = Path("plots")


def main():
    PLOTS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(INPUT_PATH)
    sns.set(style="whitegrid")

    # 10. VADER compound by domain
    plt.figure(figsize=(7, 5))
    sns.boxplot(x="domain", y="vader_compound", data=df, palette="Set2",
                order=["career", "immigration", "relationships"])
    plt.title("VADER Sentiment (Compound) by Domain")
    plt.xlabel("Domain")
    plt.ylabel("VADER Compound Score")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "10_vader_by_domain.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 11. VADER negative score by reversal, grouped by domain
    plt.figure(figsize=(8, 5))
    df["reversal_label"] = df["reversal"].map({0: "No Reversal", 1: "Reversal"})
    sns.boxplot(
        x="domain", y="vader_neg", hue="reversal_label", data=df,
        palette="Set1", order=["career", "immigration", "relationships"]
    )
    plt.title("VADER Negative Score by Reversal Status and Domain")
    plt.xlabel("Domain")
    plt.ylabel("VADER Negative Score")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "11_vader_reversal.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Regenerate time-based plots with the improved temporal data
    time_df = df[df["time_to_regret_days"].notnull()].copy()

    # Updated: Time-to-Regret histogram
    plt.figure(figsize=(7, 5))
    plt.hist(time_df["time_to_regret_days"], bins=30, edgecolor="white")
    plt.title(f"Distribution of Time to Regret (n={len(time_df)})")
    plt.xlabel("Days")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "2_time_to_regret_hist.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Updated: Time-to-Regret by domain boxplot
    plt.figure(figsize=(7, 5))
    sns.boxplot(x="domain", y="time_to_regret_days", data=time_df,
                order=["career", "immigration", "relationships"], palette="Set2")
    plt.title("Time to Regret by Domain")
    plt.xlabel("Domain")
    plt.ylabel("Days")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "6_time_to_regret_by_domain.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Updated: time availability by domain
    time_known = df.groupby("domain")["time_to_regret_days"].apply(lambda x: x.notnull().mean())
    plt.figure(figsize=(7, 5))
    time_known.reindex(["career", "immigration", "relationships"]).plot(kind="bar", color="#4C72B0")
    plt.title("Time-to-Regret Availability by Domain")
    plt.ylabel("Fraction with Time Info")
    plt.xlabel("Domain")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "9_time_availability_by_domain.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 16. Kaplan-Meier by domain
    plt.figure(figsize=(8, 5))
    kmf = KaplanMeierFitter()
    for domain in ["career", "immigration", "relationships"]:
        mask = time_df["domain"] == domain
        if mask.sum() < 10:
            continue
        kmf.fit(
            time_df.loc[mask, "time_to_regret_days"],
            event_observed=np.ones(mask.sum()),
            label=domain.capitalize(),
        )
        kmf.plot_survival_function()
    plt.title("Kaplan-Meier: Time to Regret by Domain")
    plt.xlabel("Days")
    plt.ylabel("P(regret not yet expressed)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "16_km_by_domain.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Updated: correlation heatmap with new features
    numeric_cols = ["urgency_score", "reversal", "num_comments", "score",
                    "vader_compound", "vader_neg"]
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Correlation Heatmap (with VADER)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "7_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plots to {PLOTS_DIR.absolute()}")
    print(f"  10_vader_by_domain.png")
    print(f"  11_vader_reversal.png")
    print(f"  16_km_by_domain.png")
    print(f"  Updated: 2, 6, 7, 9")


if __name__ == "__main__":
    main()

"""
Psycholinguistic features: LIWC-style word-list counts for agency, hedging,
social embeddedness, causal reasoning, and future orientation.
Adds 5 columns to all_domains_final.csv.
Saves results/psycholinguistic_summary.json, plots/26_psycholinguistic_by_domain.png.
"""
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

INPUT_PATH = Path("data_clean/all_domains_final.csv")
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")

# --- Word lists ---
AGENCY_ACTIVE = [
    "i chose", "i decided", "i wanted", "my choice", "i selected",
    "i picked", "i opted", "i made the decision", "i took",
    "it was my decision", "i initiated",
]
AGENCY_PASSIVE = [
    "was forced", "had to", "no choice", "had no option", "i was made to",
    "couldn't say no", "pressured into", "forced me", "didn't have a choice",
    "no other option", "was pushed", "was told to", "had no say",
]

HEDGING = [
    "maybe", "perhaps", "i think", "possibly", "not sure", "i guess",
    "might be", "could be", "kind of", "sort of", "somewhat",
    "i'm not certain", "i don't know if", "probably", "seemingly",
]

SOCIAL_EMBED = [
    "my wife", "my husband", "my partner", "my parents", "my kids",
    "my family", "my mom", "my dad", "my mother", "my father",
    "my brother", "my sister", "my son", "my daughter", "my boyfriend",
    "my girlfriend", "my spouse", "my children", "my friends",
    "my boss", "my coworker", "my colleague",
]

CAUSAL = [
    "because", "therefore", "so that", "as a result", "which led to",
    "caused", "due to", "consequently", "led me to", "the reason",
    "that's why", "which is why", "owing to", "on account of",
]

FUTURE = [
    "will", "going to", "plan to", "intend to", "hope to",
    "looking forward", "want to", "aiming to", "planning on",
    "considering", "thinking about", "might try",
]


def count_phrases(text: str, phrase_list: list) -> int:
    text_lower = text.lower()
    return sum(1 for phrase in phrase_list if phrase in text_lower)


def score_per_1k(count: int, word_count: int) -> float:
    if word_count == 0:
        return 0.0
    return (count / word_count) * 1000


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    text = df["full_text"].fillna("").astype(str)
    word_counts = text.str.split().str.len().clip(lower=1)

    # Compute raw counts
    agency_active = text.apply(lambda t: count_phrases(t, AGENCY_ACTIVE))
    agency_passive = text.apply(lambda t: count_phrases(t, AGENCY_PASSIVE))
    hedging_raw = text.apply(lambda t: count_phrases(t, HEDGING))
    social_raw = text.apply(lambda t: count_phrases(t, SOCIAL_EMBED))
    causal_raw = text.apply(lambda t: count_phrases(t, CAUSAL))
    future_raw = text.apply(lambda t: count_phrases(t, FUTURE))

    # Agency score: active - passive, normalized per 1k words
    df["agency_score"] = (agency_active - agency_passive).values / word_counts.values * 1000
    df["hedging_score"] = hedging_raw.values / word_counts.values * 1000
    df["social_embed_score"] = social_raw.values / word_counts.values * 1000
    df["causal_reasoning_score"] = causal_raw.values / word_counts.values * 1000
    df["future_orient_score"] = future_raw.values / word_counts.values * 1000

    df.to_csv(INPUT_PATH, index=False)
    print(f"Updated {INPUT_PATH} with 5 psycholinguistic columns")

    features = ["agency_score", "hedging_score", "social_embed_score",
                "causal_reasoning_score", "future_orient_score"]

    results = {}

    # Summary stats by domain
    for feat in features:
        domain_stats = {}
        for domain in sorted(df["domain"].unique()):
            sub = df[df["domain"] == domain][feat]
            domain_stats[domain] = {
                "mean": round(float(sub.mean()), 4),
                "std": round(float(sub.std()), 4),
                "median": round(float(sub.median()), 4),
            }
        results[f"{feat}_by_domain"] = domain_stats

    # Correlation with reversal
    corr_results = {}
    for feat in features:
        r, p = stats.pointbiserialr(df["reversal"].astype(float), df[feat].astype(float))
        corr_results[feat] = {"r": round(float(r), 4), "p_value": float(p)}
        print(f"  {feat:30s} r={r:.4f}, p={p:.4e}")
    results["correlation_with_reversal"] = corr_results

    # Kruskal-Wallis by domain for each feature
    kruskal_results = {}
    for feat in features:
        groups = [df[df["domain"] == d][feat].values for d in sorted(df["domain"].unique())]
        h_stat, p = stats.kruskal(*groups)
        kruskal_results[feat] = {"H": round(float(h_stat), 4), "p_value": float(p)}
    results["kruskal_by_domain"] = kruskal_results

    # Mann-Whitney: reversal vs non-reversal for each feature
    mw_results = {}
    for feat in features:
        rev = df[df["reversal"] == 1][feat].values
        non = df[df["reversal"] == 0][feat].values
        u, p = stats.mannwhitneyu(rev, non, alternative="two-sided")
        mw_results[feat] = {
            "U": float(u),
            "p_value": float(p),
            "rev_mean": round(float(np.mean(rev)), 4),
            "non_rev_mean": round(float(np.mean(non)), 4),
        }
    results["mannwhitney_reversal"] = mw_results

    with open(RESULTS_DIR / "psycholinguistic_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {RESULTS_DIR / 'psycholinguistic_summary.json'}")

    # --- Plot: radar-style grouped bar chart by domain ---
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
    feat_labels = ["Agency", "Hedging", "Social\nEmbeddedness", "Causal\nReasoning", "Future\nOrientation"]
    domain_colors = {"career": "#FF4500", "immigration": "#0066CC", "relationships": "#28A745"}

    for ax, feat, label in zip(axes, features, feat_labels):
        means = []
        colors = []
        labels = []
        for domain in ["career", "immigration", "relationships"]:
            sub = df[df["domain"] == domain][feat]
            means.append(sub.mean())
            colors.append(domain_colors[domain])
            labels.append(domain.capitalize())

        bars = ax.bar(range(3), means, color=colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_ylabel("Score (per 1k words)" if feat == features[0] else "")

    plt.suptitle("Psycholinguistic Features by Domain", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "26_psycholinguistic_by_domain.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {PLOTS_DIR / '26_psycholinguistic_by_domain.png'}")

    # --- Plot: features by reversal status ---
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
    rev_colors = {0: "#999999", 1: "#FF4500"}

    for ax, feat, label in zip(axes, features, feat_labels):
        for rev_val, color in rev_colors.items():
            sub = df[df["reversal"] == rev_val][feat]
            lbl = "Reversed" if rev_val == 1 else "Did not reverse"
            ax.hist(sub, bins=20, alpha=0.5, color=color, label=lbl, density=True)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.legend(fontsize=6)

    plt.suptitle("Psycholinguistic Features: Reversed vs Not", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "27_psycholinguistic_reversal.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {PLOTS_DIR / '27_psycholinguistic_reversal.png'}")


if __name__ == "__main__":
    main()

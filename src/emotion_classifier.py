"""
Emotion classification using j-hartmann/emotion-english-distilroberta-base.
7 classes: anger, disgust, fear, joy, neutral, sadness, surprise.
Reads data_clean/all_domains_final.csv, updates it with emotion/emotion_score columns.
Saves plots/18_emotion_by_domain.png, plots/19_emotion_by_reversal.png,
results/emotion_summary.json.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from transformers import pipeline

INPUT_PATH = Path("data_clean/all_domains_final.csv")
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")

EMOTION_ORDER = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


def cramers_v(confusion_matrix):
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.values.sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(INPUT_PATH)

    text = df["regret_sentence"].fillna("").astype(str)
    text = text.where(text.str.len() > 0, df["title"].fillna("").astype(str))
    sentences = text.tolist()

    print(f"Running emotion classification on {len(sentences)} posts...")
    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1,
        truncation=True,
        max_length=512,
        device=-1,
    )

    batch_size = 64
    emotions = []
    scores = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        results = classifier(batch)
        for r in results:
            top = r[0]
            emotions.append(top["label"])
            scores.append(round(top["score"], 4))
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {min(i + batch_size, len(sentences))}/{len(sentences)}")

    df["emotion"] = emotions
    df["emotion_score"] = scores
    df.to_csv(INPUT_PATH, index=False)
    print(f"Updated {INPUT_PATH} with emotion columns")

    print("\nEmotion distribution:")
    print(df["emotion"].value_counts().to_string())

    results = {}

    ct = pd.crosstab(df["emotion"], df["reversal"])
    chi2, p_val, dof, _ = stats.chi2_contingency(ct)
    cv = cramers_v(ct)
    results["chi_square_emotion_by_reversal"] = {
        "chi2": round(float(chi2), 4),
        "p_value": float(f"{p_val:.6e}"),
        "dof": int(dof),
        "cramers_v": round(float(cv), 4),
        "significant": bool(p_val < 0.05),
    }
    print(f"\nChi-square (emotion ~ reversal): chi2={chi2:.2f}, p={p_val:.4e}, V={cv:.3f}")

    ct_domain = pd.crosstab(df["emotion"], df["domain"])
    chi2_d, p_d, dof_d, _ = stats.chi2_contingency(ct_domain)
    cv_d = cramers_v(ct_domain)
    results["chi_square_emotion_by_domain"] = {
        "chi2": round(float(chi2_d), 4),
        "p_value": float(f"{p_d:.6e}"),
        "dof": int(dof_d),
        "cramers_v": round(float(cv_d), 4),
        "significant": bool(p_d < 0.05),
    }
    print(f"Chi-square (emotion ~ domain): chi2={chi2_d:.2f}, p={p_d:.4e}, V={cv_d:.3f}")

    emotion_by_domain = {}
    for domain in sorted(df["domain"].unique()):
        sub = df[df["domain"] == domain]
        dist = sub["emotion"].value_counts(normalize=True).to_dict()
        emotion_by_domain[domain] = {k: round(v, 4) for k, v in dist.items()}
    results["emotion_distribution_by_domain"] = emotion_by_domain

    reversal_by_emotion = {}
    for emo in EMOTION_ORDER:
        sub = df[df["emotion"] == emo]
        if len(sub) > 0:
            rate = float(sub["reversal"].mean())
            reversal_by_emotion[emo] = round(rate, 4)
    results["reversal_rate_by_emotion"] = reversal_by_emotion

    with open(RESULTS_DIR / "emotion_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {RESULTS_DIR}/emotion_summary.json")

    # --- Plot 18: Emotion by domain (stacked bar) ---
    domain_order = ["career", "immigration", "relationships"]
    props = pd.crosstab(df["domain"], df["emotion"], normalize="index")
    props = props.reindex(columns=EMOTION_ORDER, fill_value=0)
    props = props.reindex(domain_order)

    colors = ["#d62728", "#8c564b", "#9467bd", "#2ca02c", "#7f7f7f", "#1f77b4", "#ff7f0e"]
    fig, ax = plt.subplots(figsize=(10, 6))
    props.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Proportion")
    ax.set_xlabel("")
    ax.set_title("Emotion Distribution by Domain")
    ax.legend(title="Emotion", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xticklabels(domain_order, rotation=0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "18_emotion_by_domain.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 19: Reversal rate by emotion ---
    emo_rev = df.groupby("emotion")["reversal"].agg(["mean", "count"]).reindex(EMOTION_ORDER)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(EMOTION_ORDER)), emo_rev["mean"], color=colors, edgecolor="white")
    ax.set_xticks(range(len(EMOTION_ORDER)))
    ax.set_xticklabels(EMOTION_ORDER, rotation=30, ha="right")
    ax.set_ylabel("Reversal Rate")
    ax.set_title("Reversal Rate by Detected Emotion")
    for i, (val, n) in enumerate(zip(emo_rev["mean"], emo_rev["count"])):
        ax.text(i, val + 0.01, f"n={int(n)}", ha="center", fontsize=8, color="#555")
    ax.axhline(df["reversal"].mean(), ls="--", color="#999", lw=1, label="Overall mean")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "19_emotion_by_reversal.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plots 18, 19 to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()

"""
Cross-domain NMF topic model on all regret posts.
Saves topic keywords, updates enriched CSV with topic column,
and produces a topic-reversal heatmap.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

INPUT_PATH = Path("data_clean/all_domains_enriched.csv")
TOPICS_PATH = Path("data_clean/all_domains_topics.csv")
PLOTS_DIR = Path("plots")
N_TOPICS = 10
N_TOP_WORDS = 10


def main():
    PLOTS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(INPUT_PATH)

    text = df["regret_sentence"].fillna("").astype(str)
    text = text.where(text.str.len() > 0, df["title"].fillna("").astype(str))
    text = text.str.lower()

    tfidf = TfidfVectorizer(
        stop_words="english", max_features=10000, ngram_range=(1, 2), min_df=5
    )
    X = tfidf.fit_transform(text)
    feature_names = tfidf.get_feature_names_out()

    nmf = NMF(n_components=N_TOPICS, random_state=42, max_iter=500)
    W = nmf.fit_transform(X)

    df["topic"] = W.argmax(axis=1)

    topics = []
    for i, comp in enumerate(nmf.components_):
        top_idx = comp.argsort()[::-1][:N_TOP_WORDS]
        keywords = ", ".join(feature_names[j] for j in top_idx)
        topics.append({"topic_id": i, "keywords": keywords})

    topics_df = pd.DataFrame(topics)
    topics_df.to_csv(TOPICS_PATH, index=False)

    df.to_csv(INPUT_PATH, index=False)

    print(f"Topics saved to {TOPICS_PATH}")
    print(topics_df.to_string(index=False))
    print(f"\nTopic distribution:\n{df['topic'].value_counts().sort_index()}")

    # Per-topic reversal rate by domain
    pivot = df.groupby(["topic", "domain"])["reversal"].mean().unstack(fill_value=0)

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "Reversal Rate"},
    )
    plt.title("Reversal Rate by Topic and Domain")
    plt.xlabel("Domain")
    plt.ylabel("Topic")

    topic_labels = []
    for _, row in topics_df.iterrows():
        short = ", ".join(row["keywords"].split(", ")[:3])
        topic_labels.append(f"T{row['topic_id']}: {short}")
    plt.yticks(
        np.arange(len(topic_labels)) + 0.5,
        topic_labels,
        rotation=0,
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "12_topic_reversal_heatmap.png", dpi=150)
    plt.close()

    print(f"\nHeatmap saved to {PLOTS_DIR}/12_topic_reversal_heatmap.png")


if __name__ == "__main__":
    main()

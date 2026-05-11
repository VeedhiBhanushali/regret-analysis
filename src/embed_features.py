"""
Sentence embeddings via all-MiniLM-L6-v2, PCA reduction, UMAP visualization.
Reads data_clean/all_domains_enriched.csv.
Produces data_clean/all_domains_final.csv, data_clean/embeddings_pca50.npy,
and plots/17_umap_embeddings.png.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import umap

INPUT_PATH = Path("data_clean/all_domains_enriched.csv")
OUTPUT_PATH = Path("data_clean/all_domains_final.csv")
EMB_PATH = Path("data_clean/embeddings_pca50.npy")
PLOTS_DIR = Path("plots")

N_PCA = 50
N_PCA_KEEP = 10


def main():
    PLOTS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(INPUT_PATH)

    text = df["regret_sentence"].fillna("").astype(str)
    text = text.where(text.str.len() > 0, df["title"].fillna("").astype(str))
    sentences = text.tolist()

    print(f"Encoding {len(sentences)} sentences with all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences, show_progress_bar=True, batch_size=128)
    print(f"Raw embedding shape: {embeddings.shape}")

    pca = PCA(n_components=N_PCA, random_state=42)
    emb_pca = pca.fit_transform(embeddings)
    var_explained = pca.explained_variance_ratio_.cumsum()
    print(f"PCA {N_PCA} components explain {var_explained[-1]:.1%} of variance")
    np.save(EMB_PATH, emb_pca)
    print(f"Saved PCA embeddings to {EMB_PATH}")

    for i in range(N_PCA_KEEP):
        df[f"emb_{i}"] = emb_pca[:, i]

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH} with {N_PCA_KEEP} embedding columns")

    print("Running UMAP 2-d projection...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    emb_2d = reducer.fit_transform(emb_pca)

    domain_colors = {"career": "#FF4500", "immigration": "#0066CC", "relationships": "#28A745"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for domain, color in domain_colors.items():
        mask = df["domain"] == domain
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=color, s=4, alpha=0.4, label=domain)
    ax.set_title("UMAP of Regret Posts (by Domain)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=4, framealpha=0.9)

    ax = axes[1]
    rev_colors = {0: "#888888", 1: "#FF4500"}
    rev_labels = {0: "No reversal", 1: "Reversal"}
    for rev, color in rev_colors.items():
        mask = df["reversal"] == rev
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=color, s=4, alpha=0.4, label=rev_labels[rev])
    ax.set_title("UMAP of Regret Posts (by Reversal)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=4, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "17_umap_embeddings.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {PLOTS_DIR}/17_umap_embeddings.png")


if __name__ == "__main__":
    main()

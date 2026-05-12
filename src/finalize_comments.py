"""
Finalize comment scraping data: merge checkpoint into main dataset,
compute validation metrics, and generate plots.
Works with whatever partial data was collected.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_PATH = Path("data_clean/all_domains_final.csv")
CHECKPOINT_PATH = Path("data_clean/comments_checkpoint.csv")
OUTPUT_PATH = Path("data_clean/all_domains_comments.csv")
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    if not CHECKPOINT_PATH.exists():
        print("No checkpoint found. Run scrape_comments.py first.")
        return

    comments = pd.read_csv(CHECKPOINT_PATH)
    comments.to_csv(OUTPUT_PATH, index=False)
    print(f"Comments data: {len(comments)} posts scraped")

    # Merge into main dataset
    comments["id"] = comments["id"].astype(str)
    df["id"] = df["id"].astype(str)

    merge_cols = ["id", "upvote_ratio", "comment_vader_mean",
                  "reversal_comment_score", "n_op_replies", "n_top_comments"]
    merge_df = comments[[c for c in merge_cols if c in comments.columns]]

    # Drop existing columns if they exist to avoid duplicates
    for col in merge_cols[1:]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    merged = df.merge(merge_df, on="id", how="left")
    merged.to_csv(INPUT_PATH, index=False)
    print(f"Updated {INPUT_PATH}")

    # --- Validation analysis ---
    valid = merged[merged["reversal_comment_score"].notnull()].copy()
    valid["comment_reversal"] = (valid["reversal_comment_score"] >= 0.3).astype(int)
    n = len(valid)
    print(f"\nValidation sample: {n} posts with comment data")

    if n == 0:
        print("No valid data for validation.")
        return

    agreement = float((valid["reversal"] == valid["comment_reversal"]).mean())

    # Cohen's kappa
    ct = pd.crosstab(valid["reversal"], valid["comment_reversal"])
    if ct.shape == (2, 2):
        p_o = agreement
        p_e = (
            (ct.iloc[0].sum() / n) * (ct.sum()[0] / n)
            + (ct.iloc[1].sum() / n) * (ct.sum()[1] / n)
        )
        kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 0
    else:
        kappa = 0.0

    # Per-domain
    domain_agreement = {}
    for domain in sorted(valid["domain"].unique()):
        sub = valid[valid["domain"] == domain]
        if len(sub) > 10:
            domain_agreement[domain] = round(float((sub["reversal"] == sub["comment_reversal"]).mean()), 4)

    # Where do they disagree?
    disagree = valid[valid["reversal"] != valid["comment_reversal"]]
    keyword_only = valid[(valid["reversal"] == 1) & (valid["comment_reversal"] == 0)]
    comment_only = valid[(valid["reversal"] == 0) & (valid["comment_reversal"] == 1)]

    validation = {
        "n": n,
        "agreement_rate": round(agreement, 4),
        "cohens_kappa": round(kappa, 4),
        "confusion_matrix": ct.to_dict() if ct.shape == (2, 2) else {},
        "comment_reversal_rate": round(float(valid["comment_reversal"].mean()), 4),
        "keyword_reversal_rate": round(float(valid["reversal"].mean()), 4),
        "domain_agreement": domain_agreement,
        "n_keyword_only": int(len(keyword_only)),
        "n_comment_only": int(len(comment_only)),
        "upvote_ratio_mean": round(float(valid["upvote_ratio"].mean()), 4) if valid["upvote_ratio"].notnull().any() else None,
    }

    print(f"Agreement: {agreement:.3f}, Kappa: {kappa:.3f}")
    print(f"Confusion matrix:\n{ct}")
    print(f"Domain agreement: {domain_agreement}")

    with open(RESULTS_DIR / "comment_validation.json", "w") as f:
        json.dump(validation, f, indent=2)
    print(f"\nSaved {RESULTS_DIR / 'comment_validation.json'}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if ct.shape == (2, 2):
        ax = axes[0]
        ct_norm = ct.div(ct.sum(axis=1), axis=0)
        im = ax.imshow(ct_norm.values, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No reversal\n(comments)", "Reversal\n(comments)"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["No reversal\n(keyword)", "Reversal\n(keyword)"])
        for yi in range(2):
            for xi in range(2):
                ax.text(xi, yi, f"{ct.values[yi, xi]}\n({ct_norm.values[yi, xi]:.1%})",
                        ha="center", va="center", fontsize=10)
        ax.set_title(f"Label Agreement\n(kappa={kappa:.3f}, n={n})", fontweight="bold")
        plt.colorbar(im, ax=ax)

    ax2 = axes[1]
    if ct.shape == (2, 2):
        labels = ["Keyword only", "Comment only", "Both agree\nreversal", "Both agree\nno reversal"]
        vals = [ct.values[1, 0], ct.values[0, 1], ct.values[1, 1], ct.values[0, 0]]
        colors = ["#FF4500", "#0066CC", "#28A745", "#999999"]
        ax2.bar(range(4), vals, color=colors, alpha=0.85, edgecolor="white")
        ax2.set_xticks(range(4))
        ax2.set_xticklabels(labels, fontsize=7)
        ax2.set_ylabel("Post Count")
        ax2.set_title("Disagreement Categories", fontweight="bold")
        for i, v in enumerate(vals):
            ax2.text(i, v + 1, str(v), ha="center", fontsize=9)

    plt.suptitle("Keyword vs Comment-Based Reversal Labels", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "29_reversal_label_agreement.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {PLOTS_DIR / '29_reversal_label_agreement.png'}")


if __name__ == "__main__":
    main()

"""
Scrape comment threads for all posts in all_domains_final.csv.
Extracts: upvote_ratio, OP reply text, top comment sentiment, UPDATE signals.
Checkpoints to data_clean/comments_checkpoint.csv after every batch.
Final output: data_clean/all_domains_comments.csv, updated all_domains_final.csv,
results/comment_validation.json, plots/29_reversal_label_agreement.png.
"""
import json
import time
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

INPUT_PATH = Path("data_clean/all_domains_final.csv")
CHECKPOINT_PATH = Path("data_clean/comments_checkpoint.csv")
OUTPUT_PATH = Path("data_clean/all_domains_comments.csv")
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")

HEADERS = {"User-Agent": "RegretAnalysisSeniorProject/0.2 (public-json)"}
SLEEP_SEC = 2.5
BATCH_SIZE = 50
BATCH_PAUSE = 15

UPDATE_KEYWORDS = [
    "ended it", "broke up", "quit", "resigned", "left",
    "moved back", "filed", "withdrew", "took the job", "rejected",
    "accepted", "got divorced", "separated", "moved out", "moved on",
    "started the process", "submitted", "applied", "went back",
    "switched jobs", "gave notice", "walked away",
]

sia = SentimentIntensityAnalyzer()


def fetch_post_comments(post_id, subreddit, permalink):
    """Fetch post data + comments via Reddit JSON."""
    if permalink:
        url = f"https://www.reddit.com{permalink}.json"
    else:
        url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"

    try:
        r = requests.get(url, headers=HEADERS, params={"limit": 50, "sort": "top"},
                         timeout=15)
        if r.status_code == 429:
            print(f"  429 rate limited on {post_id}, sleeping 60s...")
            time.sleep(60)
            r = requests.get(url, headers=HEADERS, params={"limit": 50, "sort": "top"},
                             timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        if not isinstance(data, list) or len(data) < 2:
            return None
        return data
    except Exception as e:
        print(f"  Error fetching {post_id}: {e}")
        return None


def extract_comment_data(data):
    """Extract structured comment data from Reddit JSON response."""
    post = data[0]["data"]["children"][0]["data"]
    comments_raw = data[1]["data"]["children"]

    post_author = post.get("author", "")
    upvote_ratio = post.get("upvote_ratio")

    top_comments = []
    op_replies = []

    for c in comments_raw:
        if c.get("kind") != "t1":
            continue
        cd = c["data"]
        body = cd.get("body", "")
        author = cd.get("author", "")

        if author == post_author:
            op_replies.append(body)
        top_comments.append(body)

        # Also check nested replies for OP
        if "replies" in cd and isinstance(cd["replies"], dict):
            for rc in cd["replies"].get("data", {}).get("children", []):
                if rc.get("kind") == "t1" and rc["data"].get("author") == post_author:
                    op_replies.append(rc["data"].get("body", ""))

    # Comment sentiment (top 10)
    sentiments = []
    for body in top_comments[:10]:
        if body and body != "[deleted]" and body != "[removed]":
            sentiments.append(sia.polarity_scores(body)["compound"])

    comment_vader_mean = float(np.mean(sentiments)) if sentiments else np.nan

    # UPDATE/reversal signal from OP replies
    all_op_text = " ".join(op_replies).lower()
    update_hits = [kw for kw in UPDATE_KEYWORDS if kw in all_op_text]
    reversal_comment_score = min(len(update_hits) / 3.0, 1.0)

    # Also check if title has "update" in it (already in post data)
    title = post.get("title", "").lower()
    if "update" in title:
        reversal_comment_score = min(reversal_comment_score + 0.3, 1.0)

    return {
        "upvote_ratio": upvote_ratio,
        "n_top_comments": len(top_comments),
        "n_op_replies": len(op_replies),
        "comment_vader_mean": round(comment_vader_mean, 4) if not np.isnan(comment_vader_mean) else None,
        "reversal_comment_score": round(reversal_comment_score, 4),
        "update_keywords_found": ",".join(update_hits) if update_hits else "",
        "op_reply_preview": op_replies[0][:200] if op_replies else "",
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    # Load checkpoint if exists
    if CHECKPOINT_PATH.exists():
        checkpoint = pd.read_csv(CHECKPOINT_PATH)
        done_ids = set(checkpoint["id"].astype(str))
        print(f"Resuming from checkpoint: {len(done_ids)} already done")
    else:
        checkpoint = pd.DataFrame()
        done_ids = set()

    remaining = df[~df["id"].astype(str).isin(done_ids)]
    print(f"Total: {len(df)}, done: {len(done_ids)}, remaining: {len(remaining)}")

    batch_results = []
    for i, (_, row) in enumerate(remaining.iterrows()):
        post_id = str(row["id"])
        subreddit = str(row.get("subreddit", ""))
        permalink = str(row.get("permalink", ""))
        if permalink == "nan":
            permalink = ""

        data = fetch_post_comments(post_id, subreddit, permalink)
        if data:
            result = extract_comment_data(data)
            result["id"] = post_id
        else:
            result = {
                "id": post_id,
                "upvote_ratio": None,
                "n_top_comments": 0,
                "n_op_replies": 0,
                "comment_vader_mean": None,
                "reversal_comment_score": 0.0,
                "update_keywords_found": "",
                "op_reply_preview": "",
            }

        batch_results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(remaining)}")

        if (i + 1) % BATCH_SIZE == 0:
            batch_df = pd.DataFrame(batch_results)
            checkpoint = pd.concat([checkpoint, batch_df], ignore_index=True)
            checkpoint.to_csv(CHECKPOINT_PATH, index=False)
            batch_results = []
            done_ids.update(batch_df["id"].astype(str))
            print(f"  Checkpoint saved: {len(checkpoint)} total")
            print(f"  Pausing {BATCH_PAUSE}s...")
            time.sleep(BATCH_PAUSE)

        time.sleep(SLEEP_SEC)

    # Save remaining
    if batch_results:
        batch_df = pd.DataFrame(batch_results)
        checkpoint = pd.concat([checkpoint, batch_df], ignore_index=True)
        checkpoint.to_csv(CHECKPOINT_PATH, index=False)

    print(f"\nTotal comments scraped: {len(checkpoint)}")
    checkpoint.to_csv(OUTPUT_PATH, index=False)

    # Merge into main dataset
    comments_df = checkpoint[["id", "upvote_ratio", "comment_vader_mean",
                              "reversal_comment_score", "n_op_replies"]].copy()
    comments_df["id"] = comments_df["id"].astype(str)
    df["id"] = df["id"].astype(str)

    merged = df.merge(comments_df, on="id", how="left", suffixes=("", "_comment"))
    for col in ["upvote_ratio", "comment_vader_mean", "reversal_comment_score", "n_op_replies"]:
        if col + "_comment" in merged.columns:
            merged[col] = merged[col + "_comment"]
            merged.drop(columns=[col + "_comment"], inplace=True)
        elif col not in merged.columns:
            merged[col] = np.nan

    merged.to_csv(INPUT_PATH, index=False)
    print(f"Updated {INPUT_PATH} with comment features")

    # --- Reversal label validation ---
    valid = merged[merged["reversal_comment_score"].notnull()].copy()
    valid["comment_reversal"] = (valid["reversal_comment_score"] >= 0.3).astype(int)

    agreement = (valid["reversal"] == valid["comment_reversal"]).mean()
    n = len(valid)

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
        kappa = 0

    validation = {
        "n": n,
        "agreement_rate": round(agreement, 4),
        "cohens_kappa": round(kappa, 4),
        "confusion_matrix": ct.to_dict() if ct.shape == (2, 2) else {},
        "comment_reversal_rate": round(float(valid["comment_reversal"].mean()), 4),
        "keyword_reversal_rate": round(float(valid["reversal"].mean()), 4),
    }
    print(f"\nLabel validation: agreement={agreement:.3f}, kappa={kappa:.3f}")

    with open(RESULTS_DIR / "comment_validation.json", "w") as f:
        json.dump(validation, f, indent=2)
    print(f"Saved {RESULTS_DIR / 'comment_validation.json'}")

    # --- Plot: agreement matrix ---
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
        ax.set_title(f"Label Agreement (kappa={kappa:.3f})", fontweight="bold")
        plt.colorbar(im, ax=ax)

    ax2 = axes[1]
    labels = ["Keyword only", "Comment only", "Both agree\nreversal", "Both agree\nno reversal"]
    if ct.shape == (2, 2):
        vals = [ct.values[1, 0], ct.values[0, 1], ct.values[1, 1], ct.values[0, 0]]
    else:
        vals = [0, 0, 0, 0]
    colors = ["#FF4500", "#0066CC", "#28A745", "#999999"]
    ax2.bar(range(4), vals, color=colors, alpha=0.85, edgecolor="white")
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("Post Count")
    ax2.set_title("Disagreement Categories", fontweight="bold")

    plt.suptitle("Keyword vs Comment-Based Reversal Labels", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "29_reversal_label_agreement.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {PLOTS_DIR / '29_reversal_label_agreement.png'}")


if __name__ == "__main__":
    main()

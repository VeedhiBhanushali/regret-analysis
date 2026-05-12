"""
Community engagement features: uses existing num_comments/score and scraped
comment data (upvote_ratio, comment_vader_mean) to test whether community
engagement predicts reversal. Re-runs Random Forest with engagement features.
Saves results/model_comparison_v2.json, plots/31_community_features.png.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

INPUT_PATH = Path("data_clean/all_domains_final.csv")
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")


def bootstrap_auc(y_true, y_score, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    # Create engagement features from existing data
    df["comment_count_log"] = np.log1p(df["num_comments"].fillna(0))
    df["score_log"] = np.log1p(df["score"].fillna(0))

    # upvote_ratio and comment_vader_mean from scraping (many will be NaN)
    has_comment_data = df["upvote_ratio"].notnull()
    print(f"Posts with comment data: {has_comment_data.sum()}/{len(df)}")

    df.to_csv(INPUT_PATH, index=False)
    print(f"Updated {INPUT_PATH} with engagement features")

    results = {}

    # --- Correlation with reversal ---
    engagement_feats = ["comment_count_log", "score_log", "num_comments", "score"]
    if has_comment_data.sum() > 50:
        engagement_feats += ["upvote_ratio", "comment_vader_mean"]

    corr_results = {}
    for feat in engagement_feats:
        valid = df[[feat, "reversal"]].dropna()
        if len(valid) > 20:
            r, p = stats.pointbiserialr(valid["reversal"], valid[feat])
            corr_results[feat] = {"r": round(float(r), 4), "p_value": float(p)}
            print(f"  {feat:25s} r={r:+.4f}, p={p:.4e}")
    results["correlation_with_reversal"] = corr_results

    # --- Random Forest with and without engagement features ---
    base_features = ["vader_compound", "vader_neg", "urgency_score",
                     "agency_score", "hedging_score", "causal_reasoning_score",
                     "future_orient_score", "social_embed_score"]
    engagement_features = ["comment_count_log", "score_log"]

    # Model 1: base features only
    X_base = df[base_features].fillna(0).astype(float)
    y = df["reversal"].astype(int)

    clf_base = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_base = cross_val_score(clf_base, X_base, y, cv=cv, scoring="roc_auc")

    # Model 2: base + engagement
    X_full = df[base_features + engagement_features].fillna(0).astype(float)
    clf_full = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    scores_full = cross_val_score(clf_full, X_full, y, cv=cv, scoring="roc_auc")

    results["model_base"] = {
        "features": base_features,
        "mean_auc": round(float(scores_base.mean()), 4),
        "std_auc": round(float(scores_base.std()), 4),
        "cv_scores": [round(float(s), 4) for s in scores_base],
    }
    results["model_with_engagement"] = {
        "features": base_features + engagement_features,
        "mean_auc": round(float(scores_full.mean()), 4),
        "std_auc": round(float(scores_full.std()), 4),
        "cv_scores": [round(float(s), 4) for s in scores_full],
    }
    results["auc_improvement"] = round(float(scores_full.mean() - scores_base.mean()), 4)

    print(f"\nBase model AUC: {scores_base.mean():.4f} +/- {scores_base.std():.4f}")
    print(f"Engagement model AUC: {scores_full.mean():.4f} +/- {scores_full.std():.4f}")
    print(f"Improvement: {scores_full.mean() - scores_base.mean():+.4f}")

    # Feature importances (full model)
    clf_full.fit(X_full, y)
    importances = dict(zip(base_features + engagement_features, clf_full.feature_importances_))
    importances_sorted = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    results["feature_importances"] = {k: round(float(v), 4) for k, v in importances_sorted.items()}

    # --- Engagement by reversal: do high-engagement posts reverse more? ---
    df["high_engagement"] = (df["num_comments"] >= df["num_comments"].median()).astype(int)
    ct = pd.crosstab(df["high_engagement"], df["reversal"])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    results["engagement_reversal_chi2"] = {
        "chi2": round(float(chi2), 4),
        "p_value": float(p),
        "high_eng_reversal_rate": round(float(df[df["high_engagement"] == 1]["reversal"].mean()), 4),
        "low_eng_reversal_rate": round(float(df[df["high_engagement"] == 0]["reversal"].mean()), 4),
    }
    print(f"\nHigh engagement reversal: {df[df['high_engagement'] == 1]['reversal'].mean():.3f}")
    print(f"Low engagement reversal: {df[df['high_engagement'] == 0]['reversal'].mean():.3f}")
    print(f"Chi2: {chi2:.2f}, p={p:.4e}")

    with open(RESULTS_DIR / "model_comparison_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RESULTS_DIR / 'model_comparison_v2.json'}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: AUC comparison
    ax = axes[0]
    ax.bar([0, 1], [scores_base.mean(), scores_full.mean()],
           yerr=[scores_base.std(), scores_full.std()],
           color=["#999999", "#FF4500"], alpha=0.85, edgecolor="white", capsize=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Base\n(sentiment + psycholing.)", "With Engagement\n(+ comments/score)"],
                       fontsize=8)
    ax.set_ylabel("AUC (5-fold CV)")
    ax.set_title("Model Comparison", fontweight="bold")
    ax.set_ylim(0.45, 0.7)

    # Panel 2: Feature importances (top 8)
    ax2 = axes[1]
    top_feats = list(importances_sorted.items())[:8]
    feat_names = [f[0].replace("_", " ").title() for f in top_feats]
    feat_vals = [f[1] for f in top_feats]
    colors = ["#FF4500" if "log" in f[0] else "#0066CC" for f in top_feats]
    ax2.barh(range(len(feat_names)), feat_vals, color=colors, alpha=0.85, edgecolor="white")
    ax2.set_yticks(range(len(feat_names)))
    ax2.set_yticklabels(feat_names, fontsize=7)
    ax2.set_xlabel("Importance")
    ax2.set_title("Feature Importances", fontweight="bold")
    ax2.invert_yaxis()

    # Panel 3: Engagement vs reversal
    ax3 = axes[2]
    eng_bins = pd.qcut(df["num_comments"].fillna(0), q=5, duplicates="drop")
    eng_rev = df.groupby(eng_bins)["reversal"].mean()
    ax3.bar(range(len(eng_rev)), eng_rev.values, color="#FF4500", alpha=0.85, edgecolor="white")
    ax3.set_xticks(range(len(eng_rev)))
    ax3.set_xticklabels([str(x) for x in eng_rev.index], fontsize=6, rotation=30)
    ax3.set_xlabel("Comment Count (quintiles)")
    ax3.set_ylabel("Reversal Rate")
    ax3.set_title("Engagement vs Reversal", fontweight="bold")

    plt.suptitle("Community Engagement Features Analysis", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "31_community_features.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {PLOTS_DIR / '31_community_features.png'}")


if __name__ == "__main__":
    main()

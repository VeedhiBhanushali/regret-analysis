"""
Cross-domain generalization: leave-one-domain-out evaluation.
Train on 2 domains, test on held-out domain.
Reads data_clean/all_domains_final.csv.
Saves results/cross_domain.json, plots/21_cross_domain_auc.png.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

INPUT_PATH = Path("data_clean/all_domains_final.csv")
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")


def bootstrap_auc(y_true, y_score, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(y_true)
    y_true_arr = np.array(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(set(y_true_arr[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true_arr[idx], y_score[idx]))
    if len(aucs) == 0:
        return 0.5, 0.5
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    domains = sorted(df["domain"].unique())
    print(f"Domains: {domains}, total n={len(df)}")

    text = df["regret_sentence"].fillna("").astype(str)
    text = text.where(text.str.len() > 0, df["title"].fillna("").astype(str))

    num_cols = ["vader_compound", "vader_neg", "urgency_score"]
    df["has_time"] = df["time_to_regret_days"].notnull().astype(int)
    num_cols.append("has_time")

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    num_cols.extend(emb_cols)

    results = {}

    for held_out in domains:
        train_mask = df["domain"] != held_out
        test_mask = df["domain"] == held_out

        tfidf = TfidfVectorizer(stop_words="english", max_features=15000, ngram_range=(1, 2), min_df=3)
        X_text_train = tfidf.fit_transform(text[train_mask])
        X_text_test = tfidf.transform(text[test_mask])

        X_num_train = csr_matrix(df.loc[train_mask, num_cols].fillna(0).values)
        X_num_test = csr_matrix(df.loc[test_mask, num_cols].fillna(0).values)

        X_train = hstack([X_text_train, X_num_train])
        X_test = hstack([X_text_test, X_num_test])
        y_train = df.loc[train_mask, "reversal"].astype(int)
        y_test = df.loc[test_mask, "reversal"].astype(int)

        clf = RandomForestClassifier(
            n_estimators=300, max_depth=20, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]

        if len(set(y_test)) < 2:
            auc = 0.5
            ci_lo, ci_hi = 0.5, 0.5
        else:
            auc = roc_auc_score(y_test, proba)
            ci_lo, ci_hi = bootstrap_auc(y_test, proba)

        results[held_out] = {
            "test_auc": round(float(auc), 4),
            "ci_95_lower": round(ci_lo, 4),
            "ci_95_upper": round(ci_hi, 4),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "train_domains": [d for d in domains if d != held_out],
            "reversal_rate_test": round(float(y_test.mean()), 4),
        }
        print(f"Held out {held_out}: AUC={auc:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] (n_test={test_mask.sum()})")

    with open(RESULTS_DIR / "cross_domain.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RESULTS_DIR}/cross_domain.json")

    # --- Plot 21: AUC bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(domains))
    aucs = [results[d]["test_auc"] for d in domains]
    ci_los = [results[d]["ci_95_lower"] for d in domains]
    ci_his = [results[d]["ci_95_upper"] for d in domains]
    errs = [[a - lo for a, lo in zip(aucs, ci_los)], [hi - a for a, hi in zip(aucs, ci_his)]]

    colors = ["#FF4500", "#0066CC", "#28A745"]
    bars = ax.bar(x, aucs, color=colors, edgecolor="white", linewidth=0.5)
    ax.errorbar(x, aucs, yerr=errs, fmt="none", ecolor="black", capsize=6, capthick=1.5)
    ax.axhline(0.5, ls="--", color="#999", lw=1, label="Random baseline")
    ax.set_xticks(list(x))
    labels = [f"Held out: {d}\n(trained on others)" for d in domains]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Test AUC")
    ax.set_title("Cross-Domain Generalization (Leave-One-Domain-Out)")
    ax.set_ylim(0.35, 0.85)
    ax.legend()

    for i, (a, lo, hi) in enumerate(zip(aucs, ci_los, ci_his)):
        ax.text(i, a + (hi - a) + 0.02, f"{a:.3f}\n[{lo:.3f}, {hi:.3f}]",
                ha="center", fontsize=8, color="#333")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "21_cross_domain_auc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {PLOTS_DIR}/21_cross_domain_auc.png")


if __name__ == "__main__":
    main()

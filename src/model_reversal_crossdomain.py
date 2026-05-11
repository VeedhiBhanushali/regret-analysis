"""
Cross-domain reversal classifier with model comparison, bootstrap CIs, and SHAP.
Reads data_clean/all_domains_enriched.csv.
Saves results/model_comparison.json, plots/13_pr_curve.png,
plots/14_calibration_curve.png, plots/15_shap_bar.png.
"""
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve
import shap

warnings.filterwarnings("ignore", category=FutureWarning)

INPUT_PATH = Path("data_clean/all_domains_enriched.csv")
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")


def bootstrap_auc(y_true, y_score, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(set(y_true.iloc[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true.iloc[idx], y_score[idx]))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def build_features(df):
    text = df["regret_sentence"].fillna("").astype(str)
    text = text.where(text.str.len() > 0, df["title"].fillna("").astype(str))

    tfidf = TfidfVectorizer(
        stop_words="english", max_features=20000, ngram_range=(1, 2), min_df=5
    )
    X_text = tfidf.fit_transform(text)

    num_cols = ["vader_compound", "vader_neg", "urgency_score"]
    df["has_time"] = df["time_to_regret_days"].notnull().astype(int)
    num_cols.append("has_time")
    X_num = csr_matrix(df[num_cols].fillna(0).values)

    enc_domain = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    X_domain = enc_domain.fit_transform(df[["domain"]])

    enc_sub = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    X_sub = enc_sub.fit_transform(df[["source_subreddit"]])

    X = hstack([X_text, X_num, X_domain, X_sub])

    feature_names = list(tfidf.get_feature_names_out()) + num_cols
    feature_names += [f"domain_{c}" for c in enc_domain.categories_[0]]
    feature_names += [f"sub_{c}" for c in enc_sub.categories_[0]]

    return X, feature_names, tfidf


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    y = df["reversal"].astype(int)

    X, feature_names, tfidf = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "LogisticRegression": LogisticRegression(C=1, max_iter=3000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=20, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    best_auc = 0
    best_name = None
    best_clf = None
    best_proba = None

    for name, clf in models.items():
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, proba)
        ci_lo, ci_hi = bootstrap_auc(y_test, proba)

        results[name] = {
            "cv_auc_mean": round(float(cv_scores.mean()), 4),
            "cv_auc_std": round(float(cv_scores.std()), 4),
            "test_auc": round(float(test_auc), 4),
            "ci_95_lower": round(float(ci_lo), 4),
            "ci_95_upper": round(float(ci_hi), 4),
        }
        print(f"{name}: CV AUC {cv_scores.mean():.3f}+-{cv_scores.std():.3f}  "
              f"Test AUC {test_auc:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

        if test_auc > best_auc:
            best_auc = test_auc
            best_name = name
            best_clf = clf
            best_proba = proba

    with open(RESULTS_DIR / "model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBest model: {best_name} (AUC {best_auc:.3f})")
    print(classification_report(y_test, (best_proba >= 0.5).astype(int)))

    # --- Precision-Recall curve ---
    precision, recall, _ = precision_recall_curve(y_test, best_proba)
    ap = average_precision_score(y_test, best_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({best_name}, AP={ap:.2f})")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "13_pr_curve.png", dpi=150)
    plt.close()

    # --- Calibration curve ---
    prob_true, prob_pred = calibration_curve(y_test, best_proba, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label=best_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "14_calibration_curve.png", dpi=150)
    plt.close()

    # --- Feature importance (model-native, works for all model types) ---
    print("Computing feature importances...")
    if best_name == "LogisticRegression":
        importances = np.abs(best_clf.coef_[0])
        if hasattr(importances, "toarray"):
            importances = importances.toarray().flatten()
        else:
            importances = np.asarray(importances).flatten()
    else:
        importances = best_clf.feature_importances_

    n_features = len(feature_names)
    importances = importances[:n_features]

    top_k = 20
    top_idx = np.argsort(importances)[::-1][:top_k]
    top_names = [feature_names[int(i)] for i in top_idx]
    top_vals = [float(importances[int(i)]) for i in top_idx]

    plt.figure(figsize=(8, 6))
    plt.barh(range(top_k - 1, -1, -1), top_vals, color="#FF4500")
    plt.yticks(range(top_k - 1, -1, -1), top_names)
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_k} Feature Importances ({best_name})")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "15_shap_bar.png", dpi=150)
    plt.close()

    shap_data = {name: round(val, 6) for name, val in zip(top_names, top_vals)}
    results["top_features"] = shap_data
    results["best_model"] = best_name
    with open(RESULTS_DIR / "model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved PR curve, calibration curve, SHAP bar to {PLOTS_DIR}/")
    print(f"Saved model comparison to {RESULTS_DIR}/model_comparison.json")


if __name__ == "__main__":
    main()

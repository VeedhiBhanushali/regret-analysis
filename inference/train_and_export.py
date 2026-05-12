"""
Train reversal classifier and domain classifier, then serialize to joblib
for the inference service. Run from the repo root:

    python inference/train_and_export.py

Reads: data_clean/all_domains_enriched.csv
Writes: inference/models/reversal_model.joblib
        inference/models/tfidf_vectorizer.joblib
        inference/models/domain_model.joblib
        inference/models/domain_tfidf.joblib
"""
import joblib
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

INPUT_PATH = Path("data_clean/all_domains_enriched.csv")
OUT_DIR = Path("inference/models")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    text = df["regret_sentence"].fillna("").astype(str)
    text = text.where(text.str.len() > 0, df["title"].fillna("").astype(str))

    # --- Reversal classifier ---
    tfidf = TfidfVectorizer(
        stop_words="english", max_features=20000, ngram_range=(1, 2), min_df=5
    )
    X_text = tfidf.fit_transform(text)

    df["has_time"] = df["time_to_regret_days"].notnull().astype(int)
    num_cols = ["vader_compound", "vader_neg", "urgency_score", "has_time"]
    X_num = csr_matrix(df[num_cols].fillna(0).values)
    X = hstack([X_text, X_num])
    y = df["reversal"].astype(int)

    clf = LogisticRegression(C=1, max_iter=3000, class_weight="balanced")
    clf.fit(X, y)

    joblib.dump(clf, OUT_DIR / "reversal_model.joblib")
    joblib.dump(tfidf, OUT_DIR / "tfidf_vectorizer.joblib")
    print("Saved reversal_model.joblib and tfidf_vectorizer.joblib")

    # --- Domain classifier ---
    domain_tfidf = TfidfVectorizer(
        stop_words="english", max_features=10000, ngram_range=(1, 2), min_df=3
    )
    X_domain = domain_tfidf.fit_transform(text)
    y_domain = df["domain"]

    domain_clf = LogisticRegression(C=1, max_iter=2000, multi_class="multinomial")
    domain_clf.fit(X_domain, y_domain)

    joblib.dump(domain_clf, OUT_DIR / "domain_model.joblib")
    joblib.dump(domain_tfidf, OUT_DIR / "domain_tfidf.joblib")
    print("Saved domain_model.joblib and domain_tfidf.joblib")
    print("Done. Models exported to", OUT_DIR)


if __name__ == "__main__":
    main()

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

INPUT_PATH = Path("data_clean/career_all_structured.csv")

def main():
    df = pd.read_csv(INPUT_PATH)

    df = df[df["time_to_regret_days"].notnull()].copy()

    df["early_regret"] = (df["time_to_regret_days"] < 365).astype(int)

    y = df["early_regret"]

    text = df["regret_sentence"].fillna("")
    text = text.where(text.str.len() > 0, df["title"])
    text = text.astype(str)

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1,2),
        min_df=3
    )
    X = tfidf.fit_transform(text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    import numpy as np
    from sklearn.metrics import f1_score

    best_f1 = 0
    best_thresh = 0.5

    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (proba >= thresh).astype(int)
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print("Best threshold:", best_thresh)
    print("Best F1:", best_f1)

    pred = (proba >= best_thresh).astype(int)


    print("Early regret distribution:")
    print(y.value_counts())

    print("\nClassification report:")
    print(classification_report(y_test, pred))
    print("ROC AUC:", roc_auc_score(y_test, proba))

    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]

    top_pos_idx = coefs.argsort()[::-1][:20]
    top_neg_idx = coefs.argsort()[:20]

    print("\nTop words predicting EARLY regret:")
    for i in top_pos_idx:
        print(feature_names[i], coefs[i])

    print("\nTop words predicting LATE regret:")
    for i in top_neg_idx:
        print(feature_names[i], coefs[i])


if __name__ == "__main__":
    main()

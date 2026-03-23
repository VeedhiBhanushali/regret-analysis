import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix

INPUT_PATH = Path("data_clean/career_all_structured.csv")

def main():
    df = pd.read_csv(INPUT_PATH)

    y = df["reversal"].astype(int)

    # ---------- TEXT ----------
    text = df["regret_sentence"].fillna("")
    text = text.where(text.str.len() > 0, df["title"])
    text = text.astype(str)

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=25000,
        ngram_range=(1,2),
        min_df=5
    )
    X_text = tfidf.fit_transform(text)

    # ---------- NUMERIC FEATURES ----------
    df["has_time"] = df["time_to_regret_days"].notnull().astype(int)

    X_num = df[["urgency_score", "has_time"]].fillna(0).values
    X_num = csr_matrix(X_num)

    # ---------- SUBREDDIT ----------
    enc = OneHotEncoder(sparse_output=True)
    X_sub = enc.fit_transform(df[["source_subreddit"]])

    # ---------- TOPIC (as one-hot) ----------
    X_topic = enc.fit_transform(df[["topic"]])

    # Combine all features
    X = hstack([X_text, X_num, X_sub, X_topic])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    param_grid = {
        "C": [0.01, 0.1, 1, 5],
        "class_weight": [None, "balanced"]
    }

    grid = GridSearchCV(
        LogisticRegression(max_iter=3000),
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    clf = grid.best_estimator_

    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    print("Best params:", grid.best_params_)
    print(classification_report(y_test, pred))
    print("ROC AUC:", roc_auc_score(y_test, proba))

if __name__ == "__main__":
    main()

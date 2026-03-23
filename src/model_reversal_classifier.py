import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from scipy.sparse import hstack

INPUT_PATH = Path("data_clean/career_all_structured.csv")

def main():
    df = pd.read_csv(INPUT_PATH)

    # Label: reversal
    y = df["reversal"].astype(int)

    # Text: regret_sentence + title
    text = df["regret_sentence"].fillna("")
    text = text.where(text.str.len() > 0, df["title"])
    text = text.astype(str)


    tfidf = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1,2), min_df=5)
    X_text = tfidf.fit_transform(text)

    # Numeric features
    X_num = df[["urgency_score"]].fillna(0).values

    X = hstack([X_text, X_num])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, pred))
    print("ROC AUC:", roc_auc_score(y_test, proba))

    # This probability becomes the Bridge “Regret Risk Score”
    # You can demo: input text -> output risk score

if __name__ == "__main__":
    main()

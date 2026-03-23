import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

INPUT_PATH = Path("data_clean/career_all_structured.csv")
OUT_TOPICS = Path("data_clean/career_topics.csv")

N_TOPICS = 8
TOP_WORDS = 10

def main():
    df = pd.read_csv(INPUT_PATH)

    # Use regret_sentence first; fallback to title
    text = df["regret_sentence"].fillna("")
    text = text.where(text.str.len() > 0, df["title"].fillna("").astype(str))
    text = text.astype(str)

    # basic cleanup
    text = text.str.replace(r"http\S+", "", regex=True)
    text = text.str.replace(r"\s+", " ", regex=True).str.strip()

    tfidf = TfidfVectorizer(
        max_features=12000,
        stop_words="english",
        ngram_range=(1,2),
        min_df=5
    )
    X = tfidf.fit_transform(text)

    nmf = NMF(n_components=N_TOPICS, random_state=42, init="nndsvda", max_iter=400)
    W = nmf.fit_transform(X)
    H = nmf.components_

    feature_names = tfidf.get_feature_names_out()

    # Top terms per topic
    rows = []
    for k in range(N_TOPICS):
        top_idx = H[k].argsort()[::-1][:TOP_WORDS]
        terms = [feature_names[i] for i in top_idx]
        rows.append({"topic": k, "top_terms": ", ".join(terms)})

    topics_df = pd.DataFrame(rows)
    topics_df.to_csv(OUT_TOPICS, index=False)

    # Assign each post its top topic
    df["topic"] = W.argmax(axis=1)
    df.to_csv(INPUT_PATH, index=False)  # overwrite to include topic label

    print("Saved topic keywords:", OUT_TOPICS)
    print("Updated structured file with topic labels:", INPUT_PATH)
    print(topics_df)

if __name__ == "__main__":
    main()

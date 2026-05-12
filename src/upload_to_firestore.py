"""
Upload the structured regret dataset to Google Cloud Firestore.

Usage:
    export GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
    python src/upload_to_firestore.py

Reads: data_clean/all_domains_final.csv
Writes to Firestore collection: regret_posts
"""
import math
import pandas as pd
from pathlib import Path
from google.cloud import firestore

INPUT_PATH = Path("data_clean/all_domains_final.csv")
COLLECTION = "regret_posts"
BATCH_LIMIT = 450
DATABASE_ID = "regret-data"


def clean_value(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def main():
    db = firestore.Client(database=DATABASE_ID)
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    cols = [
        "id", "domain", "source_subreddit", "title", "regret_sentence",
        "vader_compound", "vader_neg", "urgency_score", "reversal",
        "time_to_regret_days", "topic", "emotion", "emotion_score",
        "event_type", "agency_score", "hedging_score",
        "social_embed_score", "causal_reasoning_score", "future_orient_score",
    ]
    available = [c for c in cols if c in df.columns]
    df_upload = df[available]

    batch = db.batch()
    count = 0
    total = 0

    for _, row in df_upload.iterrows():
        doc_id = str(row.get("id", total))
        doc_ref = db.collection(COLLECTION).document(doc_id)
        data = {k: clean_value(v) for k, v in row.to_dict().items()}
        batch.set(doc_ref, data)
        count += 1
        total += 1

        if count >= BATCH_LIMIT:
            batch.commit()
            print(f"  Committed {total} documents...")
            batch = db.batch()
            count = 0

    if count > 0:
        batch.commit()

    print(f"Uploaded {total} documents to Firestore collection '{COLLECTION}'")


if __name__ == "__main__":
    main()

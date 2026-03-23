import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

SUBREDDIT = "careeradvice"   # change later
TOTAL = 5000                 # start small; later we'll do 10,000+
PAGE_SIZE = 100             # reddit max is usually 100
SLEEP_SEC = 1.2             # be polite / avoid rate limiting

OUT_PATH = Path("data_raw")
OUT_PATH.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "RegretAnalysisSeniorProject/0.1 (by u/your_username_here)"
}

def fetch_page(after=None):
    url = f"https://www.reddit.com/r/{SUBREDDIT}/new.json"
    params = {"limit": PAGE_SIZE}
    if after:
        params["after"] = after
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def main():
    rows = []
    after = None

    while len(rows) < TOTAL:
        data = fetch_page(after=after)
        children = data.get("data", {}).get("children", [])
        if not children:
            break

        for c in children:
            d = c.get("data", {})
            rows.append({
                "id": d.get("id"),
                "subreddit": d.get("subreddit"),
                "title": d.get("title", ""),
                "selftext": d.get("selftext", ""),
                "created_utc": d.get("created_utc"),
                "created_dt": datetime.utcfromtimestamp(d["created_utc"]).isoformat() if d.get("created_utc") else None,
                "num_comments": d.get("num_comments"),
                "score": d.get("score"),
                "permalink": d.get("permalink"),
                "url": d.get("url"),
            })

            if len(rows) >= TOTAL:
                break

        after = data.get("data", {}).get("after")
        if not after:
            break

        time.sleep(SLEEP_SEC)

    df = pd.DataFrame(rows)
    out_file = OUT_PATH / f"{SUBREDDIT}_public_sample_{len(df)}.csv"
    df.to_csv(out_file, index=False)
    print(f"Saved {len(df)} rows to {out_file}")

if __name__ == "__main__":
    main()

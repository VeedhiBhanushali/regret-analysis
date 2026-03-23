import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

import sys

if len(sys.argv) < 2:
    print("Usage: python collect_reddit_buckets.py subreddit_name")
    sys.exit(1)

SUBREDDIT = sys.argv[1]


# Buckets to pull from
TOP_WINDOWS = ["day", "week", "month", "year", "all"]
ENDPOINTS = [
    ("new", "https://www.reddit.com/r/{sub}/new.json"),
    ("top", "https://www.reddit.com/r/{sub}/top.json"),
    ("hot", "https://www.reddit.com/r/{sub}/hot.json"),
]

# Regret-focused searches (high signal)
REGRET_QUERIES = [
    '"I regret"',
    '"I wish I had"',
    '"biggest mistake"',
    '"shouldn\'t have"',
    '"I messed up"',
    '"wish I never"',
    'regret',
]

PAGE_SIZE = 100
SLEEP_SEC = 2.5
MAX_PAGES_PER_BUCKET = 10  # 10 pages * 100 = 1000 max per bucket

OUT_PATH = Path("data_raw")
OUT_PATH.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "RegretAnalysisSeniorProject/0.1 (public-json)"
}

def fetch_listing(url, params, after=None):
    params = dict(params)
    params["limit"] = PAGE_SIZE
    if after:
        params["after"] = after
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def extract_rows(children):
    rows = []
    for c in children:
        d = c.get("data", {})
        if not d.get("id"):
            continue
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
    return rows

def collect_endpoint(kind, url_template):
    url = url_template.format(sub=SUBREDDIT)
    all_rows = []
    seen = set()
    after = None

    # top endpoint needs time window buckets
    if kind == "top":
        for t in TOP_WINDOWS:
            after = None
            for _ in range(MAX_PAGES_PER_BUCKET):
                data = fetch_listing(url, params={"t": t}, after=after)
                children = data.get("data", {}).get("children", [])
                if not children:
                    break
                rows = extract_rows(children)
                for r in rows:
                    if r["id"] not in seen:
                        seen.add(r["id"])
                        all_rows.append(r)
                after = data.get("data", {}).get("after")
                if not after:
                    break
                time.sleep(SLEEP_SEC)
    else:
        # new/hot
        after = None
        for _ in range(MAX_PAGES_PER_BUCKET):
            data = fetch_listing(url, params={}, after=after)
            children = data.get("data", {}).get("children", [])
            if not children:
                break
            rows = extract_rows(children)
            for r in rows:
                if r["id"] not in seen:
                    seen.add(r["id"])
                    all_rows.append(r)
            after = data.get("data", {}).get("after")
            if not after:
                break
            time.sleep(SLEEP_SEC)

    return all_rows

def collect_search():
    url = f"https://www.reddit.com/r/{SUBREDDIT}/search.json"
    all_rows = []
    seen = set()

    for q in REGRET_QUERIES:
        after = None
        for _ in range(MAX_PAGES_PER_BUCKET):
            params = {
                "q": q,
                "restrict_sr": 1,
                "sort": "new",
                "t": "all",
            }
            data = fetch_listing(url, params=params, after=after)
            children = data.get("data", {}).get("children", [])
            if not children:
                break
            rows = extract_rows(children)
            for r in rows:
                if r["id"] not in seen:
                    seen.add(r["id"])
                    all_rows.append(r)
            after = data.get("data", {}).get("after")
            if not after:
                break
            time.sleep(SLEEP_SEC)

    return all_rows

def main():
    rows = []

    for kind, url_template in ENDPOINTS:
        bucket_rows = collect_endpoint(kind, url_template)
        print(f"{kind}: collected {len(bucket_rows)}")
        rows.extend(bucket_rows)

    search_rows = collect_search()
    print(f"search: collected {len(search_rows)}")
    rows.extend(search_rows)

    # Deduplicate globally by id
    df = pd.DataFrame(rows).drop_duplicates(subset=["id"]).reset_index(drop=True)

    out_file = OUT_PATH / f"{SUBREDDIT}_buckets_{len(df)}.csv"
    df.to_csv(out_file, index=False)
    print(f"\nSaved {len(df)} unique rows to {out_file}")

if __name__ == "__main__":
    main()

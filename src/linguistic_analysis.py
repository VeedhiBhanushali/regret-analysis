"""
Linguistic analysis: log-odds ratios (Monroe et al. 2008 style), modal verb analysis.
Reads data_clean/all_domains_final.csv.
Saves results/linguistic.json, plots/22_log_odds_reversal.png.
"""
import json
import re
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_PATH = Path("data_clean/all_domains_final.csv")
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")

STOPWORDS = set(
    "i me my myself we our ours ourselves you your yours yourself he him his she her "
    "hers it its they them their theirs what which who whom this that these those am is "
    "are was were be been being have has had having do does did doing a an the and but if "
    "or because as until while of at by for with about against between through during "
    "before after above below to from up down in out on off over under again further then "
    "once here there when where why how all both each few more most other some such no nor "
    "not only own same so than too very s t can will just don should now d ll m o re ve y "
    "ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn needn shan shouldn wasn "
    "weren won wouldn would could".split()
)

MODAL_VERBS = ["should", "could", "would", "wish", "might", "must"]


def tokenize(text):
    return re.findall(r"\b[a-z]{2,}\b", text.lower())


def log_odds_ratio(counts_a, counts_b, alpha=1.0):
    """Compute log-odds with uninformative Dirichlet prior (Monroe et al. 2008 approx)."""
    vocab = set(counts_a.keys()) | set(counts_b.keys())
    n_a = sum(counts_a.values())
    n_b = sum(counts_b.values())
    n_vocab = len(vocab)

    results = {}
    for w in vocab:
        f_a = counts_a.get(w, 0) + alpha
        f_b = counts_b.get(w, 0) + alpha
        total_a = n_a + n_vocab * alpha
        total_b = n_b + n_vocab * alpha

        log_odds = np.log(f_a / total_a) - np.log(f_b / total_b)
        variance = 1.0 / f_a + 1.0 / f_b
        z_score = log_odds / np.sqrt(variance)

        if counts_a.get(w, 0) + counts_b.get(w, 0) >= 10:
            results[w] = {
                "log_odds": round(float(log_odds), 4),
                "z_score": round(float(z_score), 4),
                "freq_a": counts_a.get(w, 0),
                "freq_b": counts_b.get(w, 0),
            }
    return results


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    text = df["regret_sentence"].fillna("").astype(str)
    text = text.where(text.str.len() > 0, df["title"].fillna("").astype(str))
    df["_tokens"] = text.apply(tokenize)

    results = {}

    # --- 1. Log-odds: reversal vs non-reversal ---
    rev_tokens = [t for tokens in df.loc[df["reversal"] == 1, "_tokens"] for t in tokens if t not in STOPWORDS]
    no_rev_tokens = [t for tokens in df.loc[df["reversal"] == 0, "_tokens"] for t in tokens if t not in STOPWORDS]
    counts_rev = Counter(rev_tokens)
    counts_no_rev = Counter(no_rev_tokens)
    lo_reversal = log_odds_ratio(counts_rev, counts_no_rev)

    sorted_pro = sorted(lo_reversal.items(), key=lambda x: x[1]["z_score"], reverse=True)[:20]
    sorted_anti = sorted(lo_reversal.items(), key=lambda x: x[1]["z_score"])[:20]
    results["log_odds_reversal"] = {
        "pro_reversal_top20": {k: v for k, v in sorted_pro},
        "anti_reversal_top20": {k: v for k, v in sorted_anti},
    }
    print("Top 10 words associated with REVERSAL:")
    for w, v in sorted_pro[:10]:
        print(f"  {w}: z={v['z_score']:.2f} (rev={v['freq_a']}, no_rev={v['freq_b']})")
    print("Top 10 words associated with NO REVERSAL:")
    for w, v in sorted_anti[:10]:
        print(f"  {w}: z={v['z_score']:.2f} (rev={v['freq_a']}, no_rev={v['freq_b']})")

    # --- 2. Log-odds: early vs late regret ---
    time_df = df[df["time_to_regret_days"].notnull()].copy()
    early = time_df[time_df["time_to_regret_days"] < 180]
    late = time_df[time_df["time_to_regret_days"] > 365]

    if len(early) > 50 and len(late) > 50:
        early_tokens = [t for tokens in early["_tokens"] for t in tokens if t not in STOPWORDS]
        late_tokens = [t for tokens in late["_tokens"] for t in tokens if t not in STOPWORDS]
        lo_time = log_odds_ratio(Counter(early_tokens), Counter(late_tokens))
        sorted_early = sorted(lo_time.items(), key=lambda x: x[1]["z_score"], reverse=True)[:20]
        sorted_late = sorted(lo_time.items(), key=lambda x: x[1]["z_score"])[:20]
        results["log_odds_early_vs_late"] = {
            "early_top20": {k: v for k, v in sorted_early},
            "late_top20": {k: v for k, v in sorted_late},
        }
        print(f"\nEarly (<180d, n={len(early)}) vs Late (>365d, n={len(late)}) regret:")
        for w, v in sorted_early[:10]:
            print(f"  Early: {w}: z={v['z_score']:.2f}")
        for w, v in sorted_late[:10]:
            print(f"  Late:  {w}: z={v['z_score']:.2f}")

    # --- 3. Log-odds by domain (each vs rest) ---
    domain_lo = {}
    for domain in sorted(df["domain"].unique()):
        d_tokens = [t for tokens in df.loc[df["domain"] == domain, "_tokens"] for t in tokens if t not in STOPWORDS]
        rest_tokens = [t for tokens in df.loc[df["domain"] != domain, "_tokens"] for t in tokens if t not in STOPWORDS]
        lo = log_odds_ratio(Counter(d_tokens), Counter(rest_tokens))
        top = sorted(lo.items(), key=lambda x: x[1]["z_score"], reverse=True)[:15]
        domain_lo[domain] = {k: v for k, v in top}
        print(f"\nTop 5 distinguishing words for {domain}:")
        for w, v in top[:5]:
            print(f"  {w}: z={v['z_score']:.2f}")
    results["log_odds_by_domain"] = domain_lo

    # --- 4. Modal verb analysis ---
    modal_data = {}
    for modal in MODAL_VERBS:
        by_domain = {}
        for domain in sorted(df["domain"].unique()):
            sub = df[df["domain"] == domain]
            count = sub["_tokens"].apply(lambda t: t.count(modal)).sum()
            total = sub["_tokens"].apply(len).sum()
            rate = count / total if total > 0 else 0
            by_domain[domain] = {"count": int(count), "rate_per_1000": round(rate * 1000, 2)}

        by_reversal = {}
        for rev in [0, 1]:
            sub = df[df["reversal"] == rev]
            count = sub["_tokens"].apply(lambda t: t.count(modal)).sum()
            total = sub["_tokens"].apply(len).sum()
            rate = count / total if total > 0 else 0
            by_reversal[str(rev)] = {"count": int(count), "rate_per_1000": round(rate * 1000, 2)}

        modal_data[modal] = {"by_domain": by_domain, "by_reversal": by_reversal}
    results["modal_verbs"] = modal_data

    print("\nModal verb rates (per 1000 tokens):")
    for modal in MODAL_VERBS:
        rates = [f"{d}={modal_data[modal]['by_domain'][d]['rate_per_1000']:.1f}"
                 for d in sorted(df["domain"].unique())]
        print(f"  {modal}: {', '.join(rates)}")

    with open(RESULTS_DIR / "linguistic.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RESULTS_DIR}/linguistic.json")

    # --- Plot 22: Log-odds reversal top words ---
    top_pro = sorted_pro[:15]
    top_anti = sorted_anti[:15]
    all_words = [(w, v["z_score"]) for w, v in top_anti[::-1]] + [(w, v["z_score"]) for w, v in top_pro]

    fig, ax = plt.subplots(figsize=(9, 8))
    words = [w for w, _ in all_words]
    z_scores = [z for _, z in all_words]
    colors = ["#FF4500" if z > 0 else "#0066CC" for z in z_scores]
    y_pos = range(len(words))
    ax.barh(y_pos, z_scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(words, fontsize=8)
    ax.axvline(0, color="#333", lw=1)
    ax.set_xlabel("Z-score (log-odds ratio)")
    ax.set_title("Words Most Associated with Reversal (red) vs Non-Reversal (blue)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "22_log_odds_reversal.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {PLOTS_DIR}/22_log_odds_reversal.png")


if __name__ == "__main__":
    main()

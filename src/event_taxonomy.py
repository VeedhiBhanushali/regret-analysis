"""
Triggering event taxonomy: label each regret post with the precipitating event type.
Adds event_type column to all_domains_final.csv.
Saves results/event_taxonomy.json and plots/23_event_type_reversal.png.
"""
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

INPUT_PATH = Path("data_clean/all_domains_final.csv")
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")

# ---------------------------------------------------------------------------
# Keyword taxonomies per domain.  Priority order: first match wins.
# Each entry: (event_type_label, [substrings_to_search])
# ---------------------------------------------------------------------------

IMMIGRATION_TAXONOMY = [
    ("visa_denial_or_error", [
        "denied", "rejection", "denied my", "application denied", "visa denied",
        "gc denied", "rejected application", "rfe", "request for evidence",
        "filed wrong", "wrong form", "filing error", "mistake on application",
        "denial notice", "denial letter", "case denied",
    ]),
    ("status_or_removal", [
        "out of status", "unlawful presence", "overstay", "overstayed",
        "deported", "deportation", "removal order", "status expired",
        "expired visa", "unauthorized", "undocumented",
    ]),
    ("sponsor_or_job_loss", [
        "employer withdrew", "sponsor withdrew", "h1b cancelled", "h-1b cancelled",
        "company laid", "lost my job", "job loss", "employer pulled",
        "revoked by employer", "company went under",
    ]),
    ("voluntary_departure", [
        "moved back", "returned home", "back home", "left the country",
        "going back home", "returned to my country", "back to my home country",
        "relocated back", "went back",
    ]),
    ("application_trapped", [
        "still waiting", "years of waiting", "waiting for", "pending for",
        "in process", "case is pending", "stuck in", "backlog",
        "processing delay", "priority date",
    ]),
    ("unknown", []),  # catch-all
]

RELATIONSHIP_TAXONOMY = [
    ("abuse_or_toxicity", [
        "abuse", "abusive", "toxic", "controlling", "manipulation",
        "manipulative", "gaslighting", "harass", "threatening",
        "emotional abuse", "physically abusive", "narcissist",
    ]),
    ("infidelity", [
        "cheated", "cheating", "affair", "unfaithful", "betrayed",
        "cheat on", "was cheating",
    ]),
    ("external_pressure", [
        "family pressure", "parents forced", "forced to", "pressure from family",
        "arranged marriage", "long distance", "family disapproved",
    ]),
    ("voluntary_end", [
        "broke up", "broke it off", "ended the relationship", "ended things",
        "ended it", "left him", "left her", "left them", "divorced",
        "split up", "walked away", "cut ties",
    ]),
    ("conflict", [
        "constant fighting", "argument", "always fighting",
        "toxic fight", "disagreement", "incompatible",
    ]),
    ("unknown", []),
]

CAREER_TAXONOMY = [
    ("involuntary_exit", [
        "fired", "laid off", "layoff", "let go", "terminated",
        "pushed out", "restructur", "downsiz", "position eliminated",
        "was let go", "got fired", "got laid off",
    ]),
    ("rejected_offer", [
        "turned down the offer", "declined the offer", "rejected the offer",
        "turned it down", "said no to the offer", "rejected a job",
        "declined a job", "turned down a job", "rejected the position",
    ]),
    ("counteroffer_or_switch", [
        "accepted a counteroffer", "took a counteroffer", "switched to",
        "moved to another company", "took a different job", "jumped ship",
    ]),
    ("voluntary_quit", [
        "quit", "resigned", "gave notice", "two weeks notice",
        "left the company", "left my job", "walked out",
        "put in my notice",
    ]),
    ("promotion_or_raise_missed", [
        "missed the promotion", "passed over", "didn't get promoted",
        "no raise", "rejected for promotion", "promotion went to",
    ]),
    ("unknown", []),
]

DOMAIN_TAXONOMIES = {
    "immigration": IMMIGRATION_TAXONOMY,
    "relationships": RELATIONSHIP_TAXONOMY,
    "career": CAREER_TAXONOMY,
}


def classify_event(text: str, taxonomy: list) -> str:
    text_lower = text.lower()
    for event_type, keywords in taxonomy:
        if event_type == "unknown":
            return "unknown"
        for kw in keywords:
            if kw in text_lower:
                return event_type
    return "unknown"


def bootstrap_ci(vals, stat_fn=np.mean, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    stats_boot = [stat_fn(rng.choice(vals, len(vals), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(stats_boot, 2.5)), float(np.percentile(stats_boot, 97.5))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    text_col = (
        df["full_text"].fillna("").astype(str)
        + " "
        + df["regret_sentence"].fillna("").astype(str)
    )

    event_types = []
    for i, row in df.iterrows():
        domain = str(row.get("domain", "")).lower()
        text = str(text_col.iloc[i])
        taxonomy = DOMAIN_TAXONOMIES.get(domain, [("unknown", [])])
        event_types.append(classify_event(text, taxonomy))

    df["event_type"] = event_types
    df.to_csv(INPUT_PATH, index=False)
    print("Updated all_domains_final.csv with event_type column")

    print("\nEvent type distributions:")
    print(df.groupby(["domain", "event_type"]).size().to_string())

    results = {}

    domain_results = {}
    for domain in sorted(df["domain"].unique()):
        sub = df[df["domain"] == domain]
        print(f"\n--- {domain} ---")

        event_stats = {}
        event_vals = []
        for et in sorted(sub["event_type"].unique()):
            et_sub = sub[sub["event_type"] == et]
            n = len(et_sub)
            rev_vals = et_sub["reversal"].values.astype(float)
            rate = float(rev_vals.mean()) if n > 0 else 0.0
            if n >= 10:
                ci_lo, ci_hi = bootstrap_ci(rev_vals)
            else:
                ci_lo, ci_hi = float("nan"), float("nan")
            event_stats[et] = {
                "n": n,
                "reversal_rate": round(rate, 4),
                "ci_lower": round(ci_lo, 4) if not np.isnan(ci_lo) else None,
                "ci_upper": round(ci_hi, 4) if not np.isnan(ci_hi) else None,
            }
            event_vals.append((et, n, rate))
            print(f"  {et:30s}: n={n:4d}, reversal={rate:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]" if n >= 10
                  else f"  {et:30s}: n={n:4d}, reversal={rate:.3f} (n too small for CI)")

        # Chi-square: event_type vs reversal (exclude unknowns for cleaner test)
        known = sub[sub["event_type"] != "unknown"]
        if len(known["event_type"].unique()) >= 2 and len(known) >= 20:
            ct = pd.crosstab(known["event_type"], known["reversal"])
            try:
                chi2, p, dof, _ = stats.chi2_contingency(ct)
                event_stats["_chi_square_known"] = {
                    "chi2": round(float(chi2), 4),
                    "p_value": round(float(p), 6),
                    "dof": int(dof),
                    "n_known": int(len(known)),
                }
                print(f"  Chi-square (known event types, n={len(known)}): chi2={chi2:.2f}, p={p:.4e}")
            except Exception as e:
                print(f"  Chi-square failed: {e}")

        domain_results[domain] = event_stats

    results["event_type_by_domain"] = domain_results

    with open(RESULTS_DIR / "event_taxonomy.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RESULTS_DIR}/event_taxonomy.json")

    # --- Plot 23: grouped bar chart of reversal rate by event type, per domain ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)
    domain_colors = {
        "career": "#FF4500",
        "immigration": "#0066CC",
        "relationships": "#28A745",
    }

    for ax, domain in zip(axes, ["career", "immigration", "relationships"]):
        sub = df[df["domain"] == domain]
        stats_df = (
            sub.groupby("event_type")["reversal"]
            .agg(["mean", "count"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        stats_df = stats_df[stats_df["count"] >= 5]

        bars = ax.bar(
            range(len(stats_df)),
            stats_df["mean"],
            color=domain_colors[domain],
            alpha=0.85,
            edgecolor="white",
        )
        ax.axhline(sub["reversal"].mean(), ls="--", color="#666", lw=1, label="Domain mean")
        ax.set_xticks(range(len(stats_df)))
        ax.set_xticklabels(
            [et.replace("_", "\n") for et in stats_df["event_type"]],
            fontsize=7, ha="center"
        )
        ax.set_title(f"{domain.capitalize()}", fontweight="bold")
        ax.set_ylabel("Reversal Rate" if domain == "career" else "")
        ax.set_ylim(0, 0.7)

        for i, (_, row) in enumerate(stats_df.iterrows()):
            ax.text(i, row["mean"] + 0.01, f"n={int(row['count'])}", ha="center", fontsize=6, color="#444")

        ax.legend(fontsize=7)

    plt.suptitle("Reversal Rate by Triggering Event Type (per Domain)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "23_event_type_reversal.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {PLOTS_DIR}/23_event_type_reversal.png")


if __name__ == "__main__":
    main()

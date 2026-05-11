"""
Statistical tests with confidence intervals for the regret analysis.
Saves results/stats_summary.json.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

INPUT_PATH = Path("data_clean/all_domains_enriched.csv")
RESULTS_DIR = Path("results")


def bootstrap_ci(data, stat_fn=np.mean, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    boot_stats = []
    n = len(data)
    for _ in range(n_boot):
        sample = rng.choice(data, n, replace=True)
        boot_stats.append(stat_fn(sample))
    return float(np.percentile(boot_stats, 2.5)), float(np.percentile(boot_stats, 97.5))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(INPUT_PATH)
    results = {}

    # 1. Chi-square: reversal rate across domains
    ct = pd.crosstab(df["domain"], df["reversal"])
    chi2, p_chi, dof, _ = stats.chi2_contingency(ct)
    results["chi_square_reversal_by_domain"] = {
        "chi2": round(float(chi2), 4),
        "p_value": round(float(p_chi), 6),
        "dof": int(dof),
        "significant": bool(p_chi < 0.05),
    }
    print(f"Chi-square (reversal ~ domain): chi2={chi2:.2f}, p={p_chi:.4e}, dof={dof}")

    # 2. Bootstrap 95% CI on reversal rate per domain
    reversal_cis = {}
    for domain in sorted(df["domain"].unique()):
        vals = df.loc[df["domain"] == domain, "reversal"].values
        rate = float(vals.mean())
        ci_lo, ci_hi = bootstrap_ci(vals)
        reversal_cis[domain] = {
            "rate": round(rate, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "n": int(len(vals)),
        }
        print(f"  {domain}: reversal={rate:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] (n={len(vals)})")
    results["reversal_rate_by_domain"] = reversal_cis

    # 3. Mann-Whitney U: vader_compound for reversal=1 vs 0
    rev_0 = df.loc[df["reversal"] == 0, "vader_compound"].dropna()
    rev_1 = df.loc[df["reversal"] == 1, "vader_compound"].dropna()
    U, p_mw = stats.mannwhitneyu(rev_0, rev_1, alternative="two-sided")
    r_effect = 1 - (2 * U) / (len(rev_0) * len(rev_1))
    results["mannwhitney_vader_compound_by_reversal"] = {
        "U": round(float(U), 2),
        "p_value": round(float(p_mw), 6),
        "effect_size_r": round(float(r_effect), 4),
        "significant": bool(p_mw < 0.05),
        "mean_reversal_0": round(float(rev_0.mean()), 4),
        "mean_reversal_1": round(float(rev_1.mean()), 4),
    }
    print(f"Mann-Whitney (vader_compound ~ reversal): U={U:.0f}, p={p_mw:.4e}, r={r_effect:.3f}")

    # 4. Kruskal-Wallis H: vader_compound across domains
    groups = [g["vader_compound"].dropna().values for _, g in df.groupby("domain")]
    H, p_kw = stats.kruskal(*groups)
    results["kruskal_vader_by_domain"] = {
        "H": round(float(H), 4),
        "p_value": round(float(p_kw), 6),
        "significant": bool(p_kw < 0.05),
    }
    print(f"Kruskal-Wallis (vader_compound ~ domain): H={H:.2f}, p={p_kw:.4e}")

    # 5. Mann-Whitney on time-to-regret by reversal
    time_df = df[df["time_to_regret_days"].notnull()]
    t_rev0 = time_df.loc[time_df["reversal"] == 0, "time_to_regret_days"].values
    t_rev1 = time_df.loc[time_df["reversal"] == 1, "time_to_regret_days"].values
    if len(t_rev0) > 10 and len(t_rev1) > 10:
        U_t, p_t = stats.mannwhitneyu(t_rev0, t_rev1, alternative="two-sided")
        results["mannwhitney_time_by_reversal"] = {
            "U": round(float(U_t), 2),
            "p_value": round(float(p_t), 6),
            "significant": bool(p_t < 0.05),
            "median_reversal_0": round(float(np.median(t_rev0)), 1),
            "median_reversal_1": round(float(np.median(t_rev1)), 1),
            "n_reversal_0": int(len(t_rev0)),
            "n_reversal_1": int(len(t_rev1)),
        }
        print(f"Mann-Whitney (time ~ reversal): U={U_t:.0f}, p={p_t:.4e}")

    # 6. Bootstrap CI on median time-to-regret per domain
    time_cis = {}
    for domain in sorted(time_df["domain"].unique()):
        vals = time_df.loc[time_df["domain"] == domain, "time_to_regret_days"].values
        if len(vals) >= 10:
            med = float(np.median(vals))
            ci_lo, ci_hi = bootstrap_ci(vals, stat_fn=np.median)
            time_cis[domain] = {
                "median_days": round(med, 1),
                "ci_lower": round(ci_lo, 1),
                "ci_upper": round(ci_hi, 1),
                "n": int(len(vals)),
            }
            print(f"  {domain}: median_time={med:.0f} [{ci_lo:.0f}, {ci_hi:.0f}] (n={len(vals)})")
    results["time_to_regret_by_domain"] = time_cis

    with open(RESULTS_DIR / "stats_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR}/stats_summary.json")


if __name__ == "__main__":
    main()

"""
Propensity score matching: career vs immigration.
Tests whether the domain-reversal gap survives after matching on
observables (vader, urgency, topic, year, agency, post length).
Saves results/psm_results.json, plots/28_psm_reversal_comparison.png.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

INPUT_PATH = Path("data_clean/all_domains_final.csv")
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")


def bootstrap_ci(vals, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    means = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def nearest_neighbor_match(treated_ps, control_ps, caliper=0.05):
    """1:1 nearest-neighbor matching without replacement."""
    matches = []
    available = set(range(len(control_ps)))
    for i, ps_t in enumerate(treated_ps):
        best_j = None
        best_dist = float("inf")
        for j in available:
            dist = abs(ps_t - control_ps[j])
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is not None and best_dist <= caliper:
            matches.append((i, best_j))
            available.discard(best_j)
    return matches


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df["log_post_length"] = np.log1p(df["full_text"].fillna("").str.len())

    # Career vs Immigration
    ci_df = df[df["domain"].isin(["career", "immigration"])].copy()
    ci_df["treatment"] = (ci_df["domain"] == "immigration").astype(int)

    covariates = ["vader_compound", "vader_neg", "urgency_score",
                  "log_post_length", "post_year", "agency_score",
                  "hedging_score", "causal_reasoning_score", "future_orient_score"]

    X = ci_df[covariates].fillna(0).astype(float)
    y = ci_df["treatment"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_scaled, y)
    ci_df["propensity_score"] = lr.predict_proba(X_scaled)[:, 1]

    treated = ci_df[ci_df["treatment"] == 1].reset_index(drop=True)
    control = ci_df[ci_df["treatment"] == 0].reset_index(drop=True)

    print(f"Treated (immigration): {len(treated)}, Control (career): {len(control)}")
    print(f"Propensity scores - treated: {treated['propensity_score'].mean():.3f}, "
          f"control: {control['propensity_score'].mean():.3f}")

    matches = nearest_neighbor_match(
        treated["propensity_score"].values,
        control["propensity_score"].values,
        caliper=0.10,
    )
    print(f"Matched pairs: {len(matches)} of {len(treated)} treated units")

    if len(matches) < 20:
        print("WARNING: too few matches, relaxing caliper to 0.20")
        matches = nearest_neighbor_match(
            treated["propensity_score"].values,
            control["propensity_score"].values,
            caliper=0.20,
        )
        print(f"Matched pairs with relaxed caliper: {len(matches)}")

    # Extract matched samples
    matched_treated = treated.iloc[[m[0] for m in matches]]
    matched_control = control.iloc[[m[1] for m in matches]]

    results = {}

    # Unmatched comparison
    unmatch_rev_imm = float(ci_df[ci_df["treatment"] == 1]["reversal"].mean())
    unmatch_rev_car = float(ci_df[ci_df["treatment"] == 0]["reversal"].mean())
    results["unmatched"] = {
        "immigration_reversal": round(unmatch_rev_imm, 4),
        "career_reversal": round(unmatch_rev_car, 4),
        "gap": round(unmatch_rev_car - unmatch_rev_imm, 4),
        "n_immigration": int(len(treated)),
        "n_career": int(len(control)),
    }

    # Matched comparison
    match_rev_imm = float(matched_treated["reversal"].mean())
    match_rev_car = float(matched_control["reversal"].mean())
    att = match_rev_car - match_rev_imm

    imm_ci = bootstrap_ci(matched_treated["reversal"].values.astype(float))
    car_ci = bootstrap_ci(matched_control["reversal"].values.astype(float))

    # McNemar-style test for paired binary outcomes
    a = sum((matched_treated["reversal"].values == 1) & (matched_control["reversal"].values == 1))
    b = sum((matched_treated["reversal"].values == 1) & (matched_control["reversal"].values == 0))
    c = sum((matched_treated["reversal"].values == 0) & (matched_control["reversal"].values == 1))
    d = sum((matched_treated["reversal"].values == 0) & (matched_control["reversal"].values == 0))

    if b + c > 0:
        mcnemar_chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        mcnemar_p = float(stats.chi2.sf(mcnemar_chi2, 1))
    else:
        mcnemar_chi2, mcnemar_p = 0.0, 1.0

    results["matched"] = {
        "immigration_reversal": round(match_rev_imm, 4),
        "immigration_ci": [round(imm_ci[0], 4), round(imm_ci[1], 4)],
        "career_reversal": round(match_rev_car, 4),
        "career_ci": [round(car_ci[0], 4), round(car_ci[1], 4)],
        "att": round(att, 4),
        "n_pairs": int(len(matches)),
        "mcnemar_chi2": round(float(mcnemar_chi2), 4),
        "mcnemar_p": float(mcnemar_p),
    }

    print(f"\nUnmatched gap: {unmatch_rev_car:.3f} - {unmatch_rev_imm:.3f} = {unmatch_rev_car - unmatch_rev_imm:.3f}")
    print(f"Matched gap (ATT): {match_rev_car:.3f} - {match_rev_imm:.3f} = {att:.3f}")
    print(f"McNemar p = {mcnemar_p:.4e}")

    # Covariate balance check
    balance = {}
    for cov in covariates:
        t_vals = matched_treated[cov].fillna(0).values.astype(float)
        c_vals = matched_control[cov].fillna(0).values.astype(float)
        smd = (np.mean(t_vals) - np.mean(c_vals)) / max(
            np.sqrt((np.var(t_vals) + np.var(c_vals)) / 2), 1e-10
        )
        balance[cov] = round(float(smd), 4)
    results["covariate_balance_smd"] = balance
    print("\nCovariate balance (SMD):")
    for cov, smd in balance.items():
        flag = " ***" if abs(smd) > 0.1 else ""
        print(f"  {cov:30s}: SMD = {smd:+.4f}{flag}")

    # --- Also do career vs relationships PSM ---
    cr_df = df[df["domain"].isin(["career", "relationships"])].copy()
    cr_df["treatment"] = (cr_df["domain"] == "relationships").astype(int)

    X2 = cr_df[covariates].fillna(0).astype(float)
    y2 = cr_df["treatment"].values
    X2_scaled = scaler.fit_transform(X2)
    lr2 = LogisticRegression(max_iter=1000, random_state=42)
    lr2.fit(X2_scaled, y2)
    cr_df["propensity_score"] = lr2.predict_proba(X2_scaled)[:, 1]

    treated2 = cr_df[cr_df["treatment"] == 1].reset_index(drop=True)
    control2 = cr_df[cr_df["treatment"] == 0].reset_index(drop=True)
    matches2 = nearest_neighbor_match(
        treated2["propensity_score"].values,
        control2["propensity_score"].values,
        caliper=0.10,
    )
    if len(matches2) >= 20:
        mt2 = treated2.iloc[[m[0] for m in matches2]]
        mc2 = control2.iloc[[m[1] for m in matches2]]
        results["career_vs_relationships_matched"] = {
            "relationships_reversal": round(float(mt2["reversal"].mean()), 4),
            "career_reversal": round(float(mc2["reversal"].mean()), 4),
            "att": round(float(mc2["reversal"].mean() - mt2["reversal"].mean()), 4),
            "n_pairs": int(len(matches2)),
        }
        print(f"\nCareer vs Relationships matched: "
              f"{mc2['reversal'].mean():.3f} vs {mt2['reversal'].mean():.3f}")

    with open(RESULTS_DIR / "psm_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RESULTS_DIR / 'psm_results.json'}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Panel 1: Before vs After matching (career vs immigration)
    ax = axes[0]
    x = [0, 1]
    unmatch_vals = [unmatch_rev_car, unmatch_rev_imm]
    match_vals = [match_rev_car, match_rev_imm]
    width = 0.3
    bars1 = ax.bar([xi - width / 2 for xi in x], unmatch_vals, width, label="Unmatched",
                   color=["#FF4500", "#0066CC"], alpha=0.4, edgecolor="white")
    bars2 = ax.bar([xi + width / 2 for xi in x], match_vals, width, label="PSM Matched",
                   color=["#FF4500", "#0066CC"], alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(["Career", "Immigration"])
    ax.set_ylabel("Reversal Rate")
    ax.set_title("Career vs Immigration\n(Before & After Matching)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.55)

    for i, (u, m) in enumerate(zip(unmatch_vals, match_vals)):
        ax.text(i - width / 2, u + 0.01, f"{u:.3f}", ha="center", fontsize=8, color="#666")
        ax.text(i + width / 2, m + 0.01, f"{m:.3f}", ha="center", fontsize=8, fontweight="bold")

    # Panel 2: Propensity score distributions
    ax2 = axes[1]
    ax2.hist(treated["propensity_score"], bins=30, alpha=0.5, color="#0066CC",
             label=f"Immigration (n={len(treated)})", density=True)
    ax2.hist(control["propensity_score"], bins=30, alpha=0.5, color="#FF4500",
             label=f"Career (n={len(control)})", density=True)
    ax2.set_xlabel("Propensity Score")
    ax2.set_ylabel("Density")
    ax2.set_title("Propensity Score Distributions", fontweight="bold")
    ax2.legend(fontsize=8)

    plt.suptitle(f"Propensity Score Matching (n={len(matches)} pairs, ATT={att:+.3f})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "28_psm_reversal_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {PLOTS_DIR / '28_psm_reversal_comparison.png'}")


if __name__ == "__main__":
    main()

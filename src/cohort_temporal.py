"""
Cohort and temporal analysis: reversal rate by year, COVID era tests,
Great Resignation analysis, logistic regression with year x domain interaction,
Kaplan-Meier by era.
Saves results/cohort_analysis.json, plots/24_reversal_by_year.png, plots/25_km_by_era.png.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from lifelines import KaplanMeierFitter

INPUT_PATH = Path("data_clean/all_domains_final.csv")
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")


def bootstrap_ci(vals, stat_fn=np.mean, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    boots = [stat_fn(rng.choice(vals, len(vals), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df["reversal"] = df["reversal"].astype(int)
    df["post_year"] = df["post_year"].astype(int)

    results = {}

    # --- Reversal rate by year, per domain ---
    year_domain = (
        df.groupby(["post_year", "domain"])["reversal"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "reversal_rate", "count": "n"})
    )
    year_overall = (
        df.groupby("post_year")["reversal"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "reversal_rate", "count": "n"})
    )

    results["reversal_by_year_overall"] = {
        int(r["post_year"]): {"rate": round(r["reversal_rate"], 4), "n": int(r["n"])}
        for _, r in year_overall.iterrows()
    }

    # --- COVID era analysis ---
    df["era"] = pd.cut(
        df["post_year"],
        bins=[0, 2019, 2021, 9999],
        labels=["pre_covid", "covid", "post_covid"],
    )

    era_rates = df.groupby(["era", "domain"])["reversal"].agg(["mean", "count"]).reset_index()
    results["era_reversal_rates"] = {}
    for _, r in era_rates.iterrows():
        key = f"{r['era']}_{r['domain']}"
        results["era_reversal_rates"][key] = {
            "rate": round(float(r["mean"]), 4),
            "n": int(r["count"]),
        }

    # Chi-square: era x reversal (overall)
    ct_era = pd.crosstab(df["era"], df["reversal"])
    chi2_era, p_era, dof_era, _ = stats.chi2_contingency(ct_era)
    results["chi_square_era_reversal"] = {
        "chi2": round(float(chi2_era), 4),
        "p_value": float(p_era),
        "dof": int(dof_era),
    }
    print(f"Era x reversal chi2={chi2_era:.2f}, p={p_era:.4e}")

    # Chi-square: era x reversal per domain
    for domain in sorted(df["domain"].unique()):
        sub = df[df["domain"] == domain]
        ct = pd.crosstab(sub["era"], sub["reversal"])
        if ct.shape[0] >= 2 and ct.shape[1] >= 2:
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            results[f"chi_square_era_{domain}"] = {
                "chi2": round(float(chi2), 4),
                "p_value": float(p),
                "dof": int(dof),
            }
            print(f"  {domain}: chi2={chi2:.2f}, p={p:.4e}")

    # --- Great Resignation analysis (career, 2021-2022) ---
    career = df[df["domain"] == "career"].copy()
    career["great_resign"] = career["post_year"].isin([2021, 2022]).astype(int)

    if career["great_resign"].sum() > 10:
        gr_sub = career[career["great_resign"] == 1]
        non_gr = career[career["great_resign"] == 0]

        gr_rate = float(gr_sub["reversal"].mean())
        non_gr_rate = float(non_gr["reversal"].mean())
        gr_ci = bootstrap_ci(gr_sub["reversal"].values.astype(float))
        non_gr_ci = bootstrap_ci(non_gr["reversal"].values.astype(float))

        # Fisher exact on 2x2
        ct_gr = pd.crosstab(career["great_resign"], career["reversal"])
        odds_ratio, p_fisher = stats.fisher_exact(ct_gr)

        results["great_resignation"] = {
            "gr_reversal_rate": round(gr_rate, 4),
            "gr_n": int(len(gr_sub)),
            "gr_ci": [round(gr_ci[0], 4), round(gr_ci[1], 4)],
            "non_gr_reversal_rate": round(non_gr_rate, 4),
            "non_gr_n": int(len(non_gr)),
            "non_gr_ci": [round(non_gr_ci[0], 4), round(non_gr_ci[1], 4)],
            "odds_ratio": round(float(odds_ratio), 4),
            "p_fisher": float(p_fisher),
        }
        print(f"\nGreat Resignation (career 2021-2022): rate={gr_rate:.3f} vs {non_gr_rate:.3f}, "
              f"OR={odds_ratio:.2f}, p={p_fisher:.4e}")

    # Also check voluntary_quit specifically during Great Resignation
    if "event_type" in career.columns:
        vq = career[career["event_type"] == "voluntary_quit"].copy()
        if len(vq) > 20:
            vq_gr = vq[vq["great_resign"] == 1]
            vq_non = vq[vq["great_resign"] == 0]
            results["great_resignation_voluntary_quit"] = {
                "gr_rate": round(float(vq_gr["reversal"].mean()), 4) if len(vq_gr) > 0 else None,
                "gr_n": int(len(vq_gr)),
                "non_gr_rate": round(float(vq_non["reversal"].mean()), 4),
                "non_gr_n": int(len(vq_non)),
            }

    # --- Logistic regression: reversal ~ post_year * domain ---
    model_df = df[["reversal", "post_year", "domain"]].dropna().copy()
    model_df["year_centered"] = model_df["post_year"] - model_df["post_year"].median()

    domain_dummies = pd.get_dummies(model_df["domain"], prefix="d", drop_first=True)
    X = pd.concat([model_df[["year_centered"]], domain_dummies], axis=1)

    # Add interaction terms
    for col in domain_dummies.columns:
        X[f"year_x_{col}"] = model_df["year_centered"] * domain_dummies[col]

    X = sm.add_constant(X)
    y = model_df["reversal"]

    try:
        logit = sm.Logit(y, X.astype(float)).fit(disp=0)
        results["logistic_year_domain"] = {
            "pseudo_r2": round(float(logit.prsquared), 4),
            "aic": round(float(logit.aic), 2),
            "n": int(logit.nobs),
            "coefficients": {
                name: {
                    "coef": round(float(logit.params[name]), 4),
                    "p_value": round(float(logit.pvalues[name]), 6),
                    "ci_lower": round(float(logit.conf_int().loc[name, 0]), 4),
                    "ci_upper": round(float(logit.conf_int().loc[name, 1]), 4),
                }
                for name in logit.params.index
            },
        }
        print(f"\nLogistic regression pseudo-R2={logit.prsquared:.4f}")
        print(logit.summary2().tables[1].to_string())
    except Exception as e:
        print(f"Logistic regression failed: {e}")

    # --- Kaplan-Meier by era ---
    time_df = df[df["time_to_regret_days"].notnull()].copy()
    time_df["time_to_regret_days"] = time_df["time_to_regret_days"].astype(float)

    fig, ax = plt.subplots(figsize=(8, 5))
    era_colors = {"pre_covid": "#2196F3", "covid": "#FF5722", "post_covid": "#4CAF50"}
    era_labels = {"pre_covid": "Pre-COVID (<=2019)", "covid": "COVID (2020-2021)", "post_covid": "Post-COVID (2022+)"}

    km_results = {}
    for era_val in ["pre_covid", "covid", "post_covid"]:
        sub = time_df[time_df["era"] == era_val]
        if len(sub) < 10:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub["time_to_regret_days"], event_observed=sub["reversal"],
                label=f"{era_labels[era_val]} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, color=era_colors[era_val])
        km_results[era_val] = {
            "n": int(len(sub)),
            "median_time": round(float(sub["time_to_regret_days"].median()), 1),
            "reversal_rate": round(float(sub["reversal"].mean()), 4),
        }

    results["km_by_era"] = km_results
    ax.set_xlabel("Days to Regret Expression")
    ax.set_ylabel("Survival Probability")
    ax.set_title("Kaplan-Meier Curves by Era", fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "25_km_by_era.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {PLOTS_DIR / '25_km_by_era.png'}")

    # --- Plot: Reversal rate by year, per domain ---
    fig, ax = plt.subplots(figsize=(10, 5))
    domain_colors = {"career": "#FF4500", "immigration": "#0066CC", "relationships": "#28A745"}

    # Filter years with enough data
    for domain in ["career", "immigration", "relationships"]:
        sub = year_domain[(year_domain["domain"] == domain) & (year_domain["n"] >= 10)]
        if len(sub) < 3:
            continue
        ax.plot(sub["post_year"], sub["reversal_rate"], "o-",
                color=domain_colors[domain], label=domain.capitalize(), markersize=4, lw=1.5)

    # Overall trend line
    sub_all = year_overall[year_overall["n"] >= 20]
    ax.plot(sub_all["post_year"], sub_all["reversal_rate"], "k--", label="Overall", lw=1.5, alpha=0.5)

    # Mark COVID and Great Resignation
    ax.axvspan(2020, 2021.5, alpha=0.08, color="red", label="COVID era")
    ax.axvspan(2021, 2022.5, alpha=0.06, color="orange", label="Great Resignation")

    ax.set_xlabel("Post Year")
    ax.set_ylabel("Reversal Rate")
    ax.set_title("Reversal Rate Over Time by Domain", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0, 0.7)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "24_reversal_by_year.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {PLOTS_DIR / '24_reversal_by_year.png'}")

    # Save results
    with open(RESULTS_DIR / "cohort_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved {RESULTS_DIR / 'cohort_analysis.json'}")


if __name__ == "__main__":
    main()

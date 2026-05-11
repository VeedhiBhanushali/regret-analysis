"""
Cox Proportional Hazards regression v2 with full covariates.
Reads data_clean/all_domains_final.csv (rows with time_to_regret_days known).
Saves results/cox_v2.json, plots/20_cox_hazard_ratios.png.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from itertools import combinations

INPUT_PATH = Path("data_clean/all_domains_final.csv")
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    tdf = df[df["time_to_regret_days"].notnull()].copy()
    tdf = tdf[tdf["time_to_regret_days"] > 0]
    print(f"Cox regression on {len(tdf)} posts with valid time-to-regret")

    tdf["post_length"] = df.loc[tdf.index, "full_text"].fillna("").str.len()
    tdf["log_post_length"] = np.log1p(tdf["post_length"])

    domain_dummies = pd.get_dummies(tdf["domain"], prefix="domain", drop_first=True)
    tdf = pd.concat([tdf, domain_dummies], axis=1)

    covariates = [
        "vader_compound", "vader_neg", "urgency_score",
        "log_post_length", "topic",
    ]
    for c in domain_dummies.columns:
        covariates.append(c)

    if "emotion" in tdf.columns:
        emotion_dummies = pd.get_dummies(tdf["emotion"], prefix="emo", drop_first=True)
        tdf = pd.concat([tdf, emotion_dummies], axis=1)
        covariates.extend(emotion_dummies.columns.tolist())

    cox_df = tdf[["time_to_regret_days", "reversal"] + covariates].dropna()
    print(f"Cox model on {len(cox_df)} complete cases, {len(covariates)} covariates")

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(cox_df, duration_col="time_to_regret_days", event_col="reversal")
    cph.print_summary()

    results = {}
    results["concordance_index"] = round(float(cph.concordance_index_), 4)
    results["n_observations"] = int(len(cox_df))
    results["n_events"] = int(cox_df["reversal"].sum())

    summary = cph.summary
    hr_data = {}
    for var in summary.index:
        hr_data[var] = {
            "hazard_ratio": round(float(np.exp(summary.loc[var, "coef"])), 4),
            "coef": round(float(summary.loc[var, "coef"]), 4),
            "se": round(float(summary.loc[var, "se(coef)"]), 4),
            "p_value": round(float(summary.loc[var, "p"]), 6),
            "ci_lower": round(float(np.exp(summary.loc[var, "coef lower 95%"])), 4),
            "ci_upper": round(float(np.exp(summary.loc[var, "coef upper 95%"])), 4),
            "significant": bool(summary.loc[var, "p"] < 0.05),
        }
    results["hazard_ratios"] = hr_data

    domains = tdf["domain"].unique()
    logrank_results = {}
    for d1, d2 in combinations(sorted(domains), 2):
        g1 = tdf[tdf["domain"] == d1]
        g2 = tdf[tdf["domain"] == d2]
        lr = logrank_test(
            g1["time_to_regret_days"], g2["time_to_regret_days"],
            event_observed_A=g1["reversal"], event_observed_B=g2["reversal"],
        )
        logrank_results[f"{d1}_vs_{d2}"] = {
            "test_statistic": round(float(lr.test_statistic), 4),
            "p_value": round(float(lr.p_value), 6),
            "significant": bool(lr.p_value < 0.05),
        }
    results["logrank_tests"] = logrank_results
    print("\nLog-rank tests:")
    for k, v in logrank_results.items():
        print(f"  {k}: stat={v['test_statistic']:.2f}, p={v['p_value']:.4e}")

    with open(RESULTS_DIR / "cox_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RESULTS_DIR}/cox_v2.json")

    # --- Forest plot of hazard ratios ---
    sig_vars = [v for v in summary.index if summary.loc[v, "p"] < 0.1]
    if len(sig_vars) == 0:
        sig_vars = summary.index[:10].tolist()

    plot_vars = sig_vars[:15]
    hrs = [np.exp(summary.loc[v, "coef"]) for v in plot_vars]
    ci_lo = [np.exp(summary.loc[v, "coef lower 95%"]) for v in plot_vars]
    ci_hi = [np.exp(summary.loc[v, "coef upper 95%"]) for v in plot_vars]
    pvals = [summary.loc[v, "p"] for v in plot_vars]

    fig, ax = plt.subplots(figsize=(8, max(4, len(plot_vars) * 0.4)))
    y_pos = list(range(len(plot_vars)))
    colors = ["#FF4500" if p < 0.05 else "#999999" for p in pvals]
    for i in range(len(plot_vars)):
        ax.errorbar(hrs[i], y_pos[i], xerr=[[hrs[i] - ci_lo[i]], [ci_hi[i] - hrs[i]]],
                    fmt="o", color=colors[i], elinewidth=1.5, capsize=3, markersize=6)
    ax.axvline(1.0, ls="--", color="#666", lw=1)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(plot_vars, fontsize=9)
    ax.set_xlabel("Hazard Ratio (95% CI)")
    ax.set_title("Cox PH: Hazard Ratios for Time-to-Regret")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "20_cox_hazard_ratios.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {PLOTS_DIR}/20_cox_hazard_ratios.png")


if __name__ == "__main__":
    main()

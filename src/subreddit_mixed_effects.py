"""
Subreddit confounder analysis: compare domain-only vs subreddit-level logistic models.
Tests whether subreddit-level variation explains reversal beyond domain grouping.
Reads data_clean/all_domains_final.csv.
Saves results/subreddit_effects.json.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats
import statsmodels.api as sm

INPUT_PATH = Path("data_clean/all_domains_final.csv")
RESULTS_DIR = Path("results")


def fit_logit(y, X):
    logit = sm.Logit(y, X)
    return logit.fit(disp=0, maxiter=500, method="bfgs")


def extract_coefficients(res, names):
    coefs = {}
    ci = res.conf_int()
    for i, name in enumerate(names):
        if i >= len(res.params):
            break
        coef = float(res.params[i])
        se = float(res.bse[i])
        p = float(res.pvalues[i])
        coefs[name] = {
            "coef": round(coef, 4),
            "se": round(se, 4),
            "odds_ratio": round(float(np.exp(coef)), 4),
            "ci_lower_or": round(float(np.exp(ci[i, 0])), 4),
            "ci_upper_or": round(float(np.exp(ci[i, 1])), 4),
            "p_value": round(p, 6),
            "significant": bool(p < 0.05),
        }
    return coefs


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(INPUT_PATH)

    df["has_time"] = df["time_to_regret_days"].notnull().astype(float)
    for c in ["vader_compound", "vader_neg", "urgency_score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)

    feature_cols = ["vader_compound", "vader_neg", "urgency_score", "has_time"]

    sub_col = "subreddit" if "subreddit" in df.columns and df["subreddit"].notnull().sum() > df["source_subreddit"].notnull().sum() else "source_subreddit"
    df[sub_col] = df[sub_col].fillna("unknown")

    domain_dummies = pd.get_dummies(df["domain"], prefix="domain", drop_first=True).astype(float)
    sub_dummies = pd.get_dummies(df[sub_col], prefix="sub", drop_first=True).astype(float)

    y = df["reversal"].values.astype(float)

    print(f"Total observations: {len(df)}")
    print(f"Using subreddit column: {sub_col}")
    print(f"Subreddits: {df[sub_col].nunique()}")
    print(f"Subreddit x domain:\n{pd.crosstab(df[sub_col], df['domain']).to_string()}")

    results = {}

    # Model 1: features only (no domain/subreddit)
    X1 = sm.add_constant(df[feature_cols].values.astype(float))
    res1 = fit_logit(y, X1)
    names1 = ["const"] + feature_cols
    results["model_features_only"] = {
        "description": "reversal ~ vader + urgency + has_time",
        "coefficients": extract_coefficients(res1, names1),
        "pseudo_r2": round(float(res1.prsquared), 4),
        "aic": round(float(res1.aic), 2),
        "llf": round(float(res1.llf), 2),
    }
    print(f"\nModel 1 (features only): pseudo-R2={res1.prsquared:.4f}, AIC={res1.aic:.1f}")

    # Model 2: features + domain
    X2 = sm.add_constant(
        np.column_stack([df[feature_cols].values.astype(float), domain_dummies.values])
    )
    res2 = fit_logit(y, X2)
    names2 = ["const"] + feature_cols + domain_dummies.columns.tolist()
    results["model_with_domain"] = {
        "description": "reversal ~ vader + urgency + has_time + domain",
        "coefficients": extract_coefficients(res2, names2),
        "pseudo_r2": round(float(res2.prsquared), 4),
        "aic": round(float(res2.aic), 2),
        "llf": round(float(res2.llf), 2),
    }
    print(f"Model 2 (+ domain): pseudo-R2={res2.prsquared:.4f}, AIC={res2.aic:.1f}")

    # Model 3: features + subreddit (no domain -- subreddits subsume domain)
    X3 = sm.add_constant(
        np.column_stack([df[feature_cols].values.astype(float), sub_dummies.values])
    )
    res3 = fit_logit(y, X3)
    names3 = ["const"] + feature_cols + sub_dummies.columns.tolist()
    results["model_with_subreddit"] = {
        "description": "reversal ~ vader + urgency + has_time + subreddit",
        "coefficients": extract_coefficients(res3, names3),
        "pseudo_r2": round(float(res3.prsquared), 4),
        "aic": round(float(res3.aic), 2),
        "llf": round(float(res3.llf), 2),
    }
    print(f"Model 3 (+ subreddit): pseudo-R2={res3.prsquared:.4f}, AIC={res3.aic:.1f}")

    # LR test: Model 2 vs Model 1 (does domain help?)
    lr_12 = float(-2 * (res1.llf - res2.llf))
    df_12 = len(res2.params) - len(res1.params)
    p_12 = float(1 - sp_stats.chi2.cdf(lr_12, df_12))
    results["lr_test_domain_effect"] = {
        "lr_statistic": round(lr_12, 4),
        "df": df_12,
        "p_value": round(p_12, 6),
        "significant": bool(p_12 < 0.05),
        "interpretation": "Does adding domain improve model over features alone?",
    }
    print(f"\nLR test (domain effect): stat={lr_12:.2f}, df={df_12}, p={p_12:.4e}")

    # LR test: Model 3 vs Model 2 (does subreddit add beyond domain?)
    lr_23 = float(-2 * (res2.llf - res3.llf))
    df_23 = len(res3.params) - len(res2.params)
    p_23 = float(1 - sp_stats.chi2.cdf(lr_23, df_23))
    results["lr_test_subreddit_beyond_domain"] = {
        "lr_statistic": round(lr_23, 4),
        "df": df_23,
        "p_value": round(p_23, 6),
        "significant": bool(p_23 < 0.05),
        "interpretation": "Does subreddit-level variation add predictive power beyond domain?",
    }
    print(f"LR test (subreddit beyond domain): stat={lr_23:.2f}, df={df_23}, p={p_23:.4e}")

    # Compare feature coefficient stability across models
    stability = {}
    for feat in feature_cols:
        idx1 = names1.index(feat) if feat in names1 else None
        idx2 = names2.index(feat) if feat in names2 else None
        idx3 = names3.index(feat) if feat in names3 else None
        if idx1 is not None and idx2 is not None and idx3 is not None:
            stability[feat] = {
                "or_features_only": round(float(np.exp(res1.params[idx1])), 4),
                "or_with_domain": round(float(np.exp(res2.params[idx2])), 4),
                "or_with_subreddit": round(float(np.exp(res3.params[idx3])), 4),
            }
    results["coefficient_stability"] = stability
    print("\nCoefficient stability (odds ratios):")
    for feat, vals in stability.items():
        print(f"  {feat}: {vals['or_features_only']:.3f} -> {vals['or_with_domain']:.3f} -> {vals['or_with_subreddit']:.3f}")

    results["n_observations"] = int(len(df))
    results["n_subreddits"] = int(df[sub_col].nunique())
    results["n_domains"] = int(df["domain"].nunique())

    with open(RESULTS_DIR / "subreddit_effects.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RESULTS_DIR}/subreddit_effects.json")


if __name__ == "__main__":
    main()

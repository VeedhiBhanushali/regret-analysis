"""
LLM-based structured annotation of a stratified 500-post sample.
Uses GPT-4o-mini for structured extraction: decision type, trigger,
reversal confidence, agency level, emotional clarity.
Computes Cohen's kappa vs keyword-based reversal label.
Saves results/llm_annotations.json, results/llm_validation_summary.json,
plots/30_llm_vs_keyword_confusion.png.
"""
import json
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from openai import OpenAI

INPUT_PATH = Path("data_clean/all_domains_final.csv")
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")
ANNOTATIONS_PATH = RESULTS_DIR / "llm_annotations.json"

SAMPLE_SIZE = 500
BATCH_SIZE = 10

SYSTEM_PROMPT = """You are a research annotator for a regret analysis study. 
For each Reddit post about regret, extract structured information.
Respond ONLY with valid JSON, no markdown formatting."""

USER_TEMPLATE = """Analyze this Reddit post about regret and extract:

Title: {title}
Text: {text}

Return a JSON object with these exact fields:
{{
  "decision_type": "the type of decision (e.g., career change, breakup, immigration move)",
  "specific_trigger": "what specifically triggered the regret (1 sentence max)",
  "reversal_confident": true or false (did the person clearly act to reverse their decision?),
  "reversal_direction": "acted" or "did not act" or "unclear",
  "agency_level": "high" or "medium" or "low" (did the person choose this or was it forced?),
  "emotional_clarity": "high" or "medium" or "low" (how clear is the emotional expression?)
}}"""


def annotate_post(client, title, text, max_retries=2):
    text_truncated = text[:1500] if len(text) > 1500 else text
    prompt = USER_TEMPLATE.format(title=title, text=text_truncated)

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except (json.JSONDecodeError, Exception) as e:
            if attempt < max_retries:
                time.sleep(2)
                continue
            return {"error": str(e)}


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    client = OpenAI()

    df = pd.read_csv(INPUT_PATH)

    # Stratified sample: balanced by domain and reversal
    sample_frames = []
    per_stratum = SAMPLE_SIZE // 6  # 3 domains x 2 reversal values

    for domain in sorted(df["domain"].unique()):
        for rev in [0, 1]:
            stratum = df[(df["domain"] == domain) & (df["reversal"] == rev)]
            n = min(per_stratum, len(stratum))
            sample_frames.append(stratum.sample(n=n, random_state=42))

    sample = pd.concat(sample_frames, ignore_index=True).sample(frac=1, random_state=42)
    print(f"Sample size: {len(sample)}")
    print(f"By domain: {sample['domain'].value_counts().to_dict()}")
    print(f"By reversal: {sample['reversal'].value_counts().to_dict()}")

    # Load existing annotations if resuming
    if ANNOTATIONS_PATH.exists():
        with open(ANNOTATIONS_PATH) as f:
            existing = json.load(f)
        done_ids = {a["id"] for a in existing}
        annotations = existing
        print(f"Resuming: {len(done_ids)} already annotated")
    else:
        done_ids = set()
        annotations = []

    remaining = sample[~sample["id"].astype(str).isin(done_ids)]
    print(f"Remaining to annotate: {len(remaining)}")

    for i, (_, row) in enumerate(remaining.iterrows()):
        title = str(row.get("title", ""))
        text = str(row.get("full_text", ""))
        post_id = str(row["id"])

        result = annotate_post(client, title, text)
        result["id"] = post_id
        result["domain"] = row["domain"]
        result["keyword_reversal"] = int(row["reversal"])
        annotations.append(result)

        if (i + 1) % BATCH_SIZE == 0:
            with open(ANNOTATIONS_PATH, "w") as f:
                json.dump(annotations, f, indent=2)
            print(f"  Annotated {i + 1}/{len(remaining)} (saved checkpoint)")

        time.sleep(0.5)

    with open(ANNOTATIONS_PATH, "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"\nSaved {len(annotations)} annotations to {ANNOTATIONS_PATH}")

    # --- Validation analysis ---
    valid_annotations = [a for a in annotations if "error" not in a]
    print(f"Valid annotations: {len(valid_annotations)} / {len(annotations)}")

    llm_reversal = []
    keyword_reversal = []
    domains = []
    agency_levels = []

    for a in valid_annotations:
        llm_rev = 1 if a.get("reversal_confident") is True else 0
        llm_reversal.append(llm_rev)
        keyword_reversal.append(a["keyword_reversal"])
        domains.append(a["domain"])
        agency_levels.append(a.get("agency_level", "unknown"))

    llm_arr = np.array(llm_reversal)
    kw_arr = np.array(keyword_reversal)

    agreement = float(np.mean(llm_arr == kw_arr))

    # Cohen's kappa
    ct = pd.crosstab(
        pd.Series(kw_arr, name="keyword"),
        pd.Series(llm_arr, name="llm"),
    )
    n = len(llm_arr)
    if ct.shape == (2, 2):
        p_o = agreement
        p_e = (
            (ct.iloc[0].sum() / n) * (ct.sum()[0] / n)
            + (ct.iloc[1].sum() / n) * (ct.sum()[1] / n)
        )
        kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 0
    else:
        p_e = 0
        kappa = 0

    # Per-domain agreement
    domain_agreement = {}
    for domain in sorted(set(domains)):
        mask = np.array([d == domain for d in domains])
        if mask.sum() > 0:
            domain_agreement[domain] = round(float(np.mean(llm_arr[mask] == kw_arr[mask])), 4)

    # Agency distribution
    agency_dist = pd.Series(agency_levels).value_counts().to_dict()

    summary = {
        "n_valid": len(valid_annotations),
        "n_errors": len(annotations) - len(valid_annotations),
        "overall_agreement": round(agreement, 4),
        "cohens_kappa": round(kappa, 4),
        "confusion_matrix": ct.to_dict() if ct.shape == (2, 2) else {},
        "domain_agreement": domain_agreement,
        "llm_reversal_rate": round(float(llm_arr.mean()), 4),
        "keyword_reversal_rate": round(float(kw_arr.mean()), 4),
        "agency_distribution": agency_dist,
        "reversal_direction_dist": pd.Series([
            a.get("reversal_direction", "unknown") for a in valid_annotations
        ]).value_counts().to_dict(),
    }

    with open(RESULTS_DIR / "llm_validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nValidation: agreement={agreement:.3f}, kappa={kappa:.3f}")
    print(f"Saved {RESULTS_DIR / 'llm_validation_summary.json'}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Confusion matrix
    if ct.shape == (2, 2):
        ax = axes[0]
        ct_norm = ct.div(ct.sum(axis=1), axis=0)
        im = ax.imshow(ct_norm.values, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No reversal\n(LLM)", "Reversal\n(LLM)"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["No reversal\n(keyword)", "Reversal\n(keyword)"])
        for yi in range(2):
            for xi in range(2):
                ax.text(xi, yi, f"{ct.values[yi, xi]}\n({ct_norm.values[yi, xi]:.1%})",
                        ha="center", va="center", fontsize=10)
        ax.set_title(f"Confusion Matrix\n(kappa={kappa:.3f})", fontweight="bold")

    # Per-domain agreement
    ax2 = axes[1]
    domain_colors = {"career": "#FF4500", "immigration": "#0066CC", "relationships": "#28A745"}
    for i, (d, rate) in enumerate(sorted(domain_agreement.items())):
        ax2.bar(i, rate, color=domain_colors.get(d, "#999"), alpha=0.85, edgecolor="white")
        ax2.text(i, rate + 0.01, f"{rate:.1%}", ha="center", fontsize=9)
    ax2.set_xticks(range(len(domain_agreement)))
    ax2.set_xticklabels([d.capitalize() for d in sorted(domain_agreement.keys())])
    ax2.set_ylabel("Agreement Rate")
    ax2.set_ylim(0, 1)
    ax2.set_title("Agreement by Domain", fontweight="bold")

    # Agency distribution
    ax3 = axes[2]
    agency_order = ["high", "medium", "low"]
    agency_vals = [agency_dist.get(a, 0) for a in agency_order]
    ax3.bar(range(3), agency_vals, color=["#28A745", "#FFC107", "#FF4500"],
            alpha=0.85, edgecolor="white")
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(["High", "Medium", "Low"])
    ax3.set_ylabel("Count")
    ax3.set_title("Agency Level (LLM)", fontweight="bold")

    plt.suptitle(f"LLM Annotation Validation (n={len(valid_annotations)})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "30_llm_vs_keyword_confusion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {PLOTS_DIR / '30_llm_vs_keyword_confusion.png'}")


if __name__ == "__main__":
    main()

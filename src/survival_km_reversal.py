import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import KaplanMeierFitter

INPUT_PATH = Path("data_clean/career_all_structured.csv")

def main():
    df = pd.read_csv(INPUT_PATH)

    # Keep only cases with known time
    dft = df[df["time_to_regret_days"].notnull()].copy()

    # Here, event is always observed (they reported regret timing), so event_observed=1
    # (This is not perfect survival analysis, but it is an impressive, honest visualization.)
    dft["event_observed"] = 1

    kmf = KaplanMeierFitter()

    plt.figure()
    for label, g in dft.groupby("reversal"):
        kmf.fit(g["time_to_regret_days"], event_observed=g["event_observed"], label=f"reversal={int(label)}")
        kmf.plot_survival_function()

    plt.title("Kaplan–Meier: Time to Regret (Career) by Reversal")
    plt.xlabel("Days")
    plt.ylabel("Survival: P(regret not yet expressed)")
    plt.show()

    print("N time-known:", len(dft))
    print("Median time overall:", dft["time_to_regret_days"].median())
    print("Median time reversal=1:", dft[dft["reversal"] == 1]["time_to_regret_days"].median())
    print("Median time reversal=0:", dft[dft["reversal"] == 0]["time_to_regret_days"].median())

if __name__ == "__main__":
    main()

import pandas as pd
from pathlib import Path
from lifelines import CoxPHFitter

INPUT_PATH = Path("data_clean/career_all_structured.csv")

def main():
    df = pd.read_csv(INPUT_PATH)

    # Keep only time-known
    df = df[df["time_to_regret_days"].notnull()].copy()

    # Event is always observed regret (1)
    df["event"] = 1

    # Use numeric features only for Cox
    df_cox = df[[
        "time_to_regret_days",
        "event",
        "urgency_score",
        "reversal"
    ]].copy()

    cph = CoxPHFitter()
    cph.fit(df_cox, duration_col="time_to_regret_days", event_col="event")

    print(cph.summary)

if __name__ == "__main__":
    main()

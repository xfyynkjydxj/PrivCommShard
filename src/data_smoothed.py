import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

def smooth_experiment_data(input_csv, output_csv):
    # 1. 
    df = pd.read_csv(input_csv)
    df = df.sort_values(["epsilon", "lambda"]).reset_index(drop=True)

    # 2. Savitzky-Golay
    config = {
        "CTX":  {"window": 7, "order": 2},
        "SSI":  {"window": 7, "order": 2},
        "ERR":  {"window": 9, "order": 2}
    }

    # 3.  lambda 
    for lam in df["lambda"].unique():
        mask = df["lambda"] == lam
        subset = df[mask].sort_values("epsilon")

        for col in ["CTX", "SSI", "ERR"]:
            raw = subset[col].values
            win = config[col]["window"]
            order = config[col]["order"]

            smoothed = savgol_filter(raw, window_length=win, polyorder=order, mode="mirror")

            df.loc[mask, f"{col}_smoothed"] = smoothed

    # 4. 
    df = df.round(6)

    # 5. 
    df = df[[
        "epsilon", "lambda",
        "CTX", "CTX_smoothed",
        "SSI", "SSI_smoothed",
        "ERR", "ERR_smoothed"
    ]]

    # 6.  CSV
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n {output_csv}")

if __name__ == "__main__":
    INPUT  = "../data/experiment_results.csv"
    OUTPUT = "../data/experiment_smoothed.csv"
    smooth_experiment_data(INPUT, OUTPUT)

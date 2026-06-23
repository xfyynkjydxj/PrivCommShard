import pandas as pd
import numpy as np
from scipy.stats import spearmanr

SHARD_CONFIG = {
    4: "../data/nmi_results4.csv",
    8: "../data/nmi_results8.csv",
    16: "../data/nmi_results16.csv",
    32: "../data/nmi_results32.csv",
    64: "../data/nmi_results64.csv"
}

# 1. 
def load(csv_path):
    df = pd.read_csv(csv_path)
    if "ERR(RA)" in df.columns:
        df = df.rename(columns={"ERR(RA)": "ESR"})
    df = df.drop_duplicates(subset=["epsilon", "lambda"])
    return df

# 2. 
def monotonicity(x, y):
    # Spearman  + direction check
    rho, _ = spearmanr(x, y)

    diffs = np.diff(y)
    increasing_ratio = np.mean(diffs >= 0)

    return {
        "spearman_rho": rho,
        "increasing_ratio": increasing_ratio
    }

# 3. 1
def check_epsilon_trend(df):
    #  lambda
    df_fix = df.groupby("epsilon").mean(numeric_only=True).reset_index()

    nmi_stat = monotonicity(df_fix["epsilon"], df_fix["NMI"])
    esr_stat = monotonicity(df_fix["epsilon"], df_fix["ESR"])

    nmi_slope = np.mean(np.abs(np.diff(df_fix["NMI"])))

    return {
        "NMI_vs_eps": nmi_stat,
        "ESR_vs_eps": esr_stat,
        "NMI_flatness": nmi_slope
    }

# 4. 2
def check_lambda_effect(df):
    df_fix = df.groupby("lambda").mean(numeric_only=True).reset_index()

    nmi_var = np.var(df_fix["NMI"])
    esr_var = np.var(df_fix["ESR"])
    ctx_var = np.var(df_fix["CTX"]) if "CTX" in df.columns else None

    corr_nmi = spearmanr(df_fix["lambda"], df_fix["NMI"])[0]
    corr_esr = spearmanr(df_fix["lambda"], df_fix["ESR"])[0]

    return {
        "NMI_variance": nmi_var,
        "ESR_variance": esr_var,
        "CTX_variance": ctx_var,
        "corr_lambda_NMI": corr_nmi,
        "corr_lambda_ESR": corr_esr
    }

# 5. Shard scale
def check_scale_consistency(results):
    nmi_rhos = [v["eps"]["NMI_vs_eps"]["spearman_rho"] for v in results.values()]
    esr_rhos = [v["eps"]["ESR_vs_eps"]["spearman_rho"] for v in results.values()]

    return {
        "NMI_rho_mean": np.mean(nmi_rhos),
        "NMI_rho_std": np.std(nmi_rhos),
        "ESR_rho_mean": np.mean(esr_rhos),
        "ESR_rho_std": np.std(esr_rhos),
    }

# 6. 
def main():
    results = {}

    print("\n========== VALIDATION REPORT ==========\n")

    for shard, path in SHARD_CONFIG.items():
        df = load(path)

        eps_check = check_epsilon_trend(df)
        lam_check = check_lambda_effect(df)

        results[shard] = {
            "eps": eps_check,
            "lambda": lam_check
        }

        print(f"\n--- ShardNum = {shard} ---")

        print("[  NMI]")
        print(eps_check["NMI_vs_eps"])

        print("[  ESR]")
        print(eps_check["ESR_vs_eps"])

        print(f"NMI flatness (mean ||): {eps_check['NMI_flatness']:.6f}")

        print("[ effect]")
        print(f"corr(, NMI) = {lam_check['corr_lambda_NMI']:.3f}")
        print(f"corr(, ESR) = {lam_check['corr_lambda_ESR']:.3f}")

    # scale invariance summary
    scale_stats = check_scale_consistency(results)

    print("\n========== SCALE INVARIANCE ==========")
    print(scale_stats)

    print("\n========== INTERPRETATION GUIDE ==========")
    print(" > 0.7 => strong monotonic trend")
    print("corr  0 =>  is secondary factor")
    print("low std across shards => scale invariance supported")

if __name__ == "__main__":
    main()

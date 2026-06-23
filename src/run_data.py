import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

SHARD_CONFIG = {
    4: "../data/nmi_results4.csv",
    8: "../data/nmi_results8.csv",
    16: "../data/nmi_results16.csv",
    32: "../data/nmi_results32.csv",
    64: "../data/nmi_results64.csv"
}

def load(csv_path):
    df = pd.read_csv(csv_path)
    if "ERR(RA)" in df.columns:
        df = df.rename(columns={"ERR(RA)": "ESR"})
    df = df.drop_duplicates(subset=["epsilon", "lambda"])
    return df

def build_surface(df, metric):
    eps = sorted(df["epsilon"].unique())
    lam = sorted(df["lambda"].unique())

    X, Y = np.meshgrid(eps, lam)
    Z = np.zeros_like(X, dtype=float)

    for i, l in enumerate(lam):
        for j, e in enumerate(eps):
            val = df[(df["epsilon"] == e) & (df["lambda"] == l)][metric]
            if len(val) > 0:
                Z[i, j] = val.values[0]

    return X, Y, Z

# 25 SCI
def plot_all_surfaces(all_data):

    shards = list(all_data.keys())

    fig = plt.figure(figsize=(22, 8))

    # ===================== NMI =====================
    for idx, shard in enumerate(shards):
        df = all_data[shard]
        X, Y, Z = build_surface(df, "NMI")

        ax = fig.add_subplot(2, 5, idx + 1, projection='3d')

        ax.plot_surface(
            X, Y, Z,
            cmap="viridis",
            alpha=0.92,
            edgecolor='k',
            linewidth=0.15,
            antialiased=True
        )

        ax.view_init(elev=35, azim=-120)

        ax.set_title(f"NMI | Shard={shard}", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("NMI")

    # ===================== ESR =====================
    for idx, shard in enumerate(shards):
        df = all_data[shard]
        X, Y, Z = build_surface(df, "ESR")

        ax = fig.add_subplot(2, 5, 5 + idx + 1, projection='3d')

        ax.plot_surface(
            X, Y, Z,
            cmap="coolwarm",
            alpha=0.92,
            edgecolor='k',
            linewidth=0.15,
            antialiased=True
        )

        ax.view_init(elev=35, azim=-120)

        ax.set_title(f"ESR | Shard={shard}", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("ESR")

    plt.tight_layout()
    plt.savefig("ALL_SHARD_2x5_SURFACE.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":

    all_data = {}

    for shard, path in SHARD_CONFIG.items():
        all_data[shard] = load(path)

    plot_all_surfaces(all_data)

    print(" Saved: ALL_SHARD_2x5_SURFACE.png")

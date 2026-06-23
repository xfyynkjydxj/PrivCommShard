import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_3d_surface(df, z_col, title, z_label, save_name):
    eps_vals = df['epsilon'].unique()
    lam_vals = df['lambda'].unique()
    X, Y = np.meshgrid(eps_vals, lam_vals)

    #  Z 
    Z = np.zeros((len(lam_vals), len(eps_vals)))
    for i, lam in enumerate(lam_vals):
        sub = df[df['lambda'] == lam]
        Z[i, :] = sub[z_col].values

    #  3D 
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        X, Y, Z,
        cmap='viridis',
        edgecolor='none',
        rstride=2, cstride=2,
        antialiased=True
    )

    ax.set_xlabel(' (Privacy Budget)', fontsize=12, labelpad=10)
    ax.set_ylabel(' (Temporal Weight)', fontsize=12, labelpad=10)
    ax.set_zlabel(z_label, fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=15)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" {save_name}")

if __name__ == "__main__":
    df = pd.read_csv("../data/experiment_smoothed.csv")

    #  3  3D 
    plot_3d_surface(df, "CTX", "CTX", "CTX", "3D_CTX.png")
    plot_3d_surface(df, "SSI", "SSI", "SSI", "3D_SSI.png")
    plot_3d_surface(df, "ERR", "ERR", "ERR", "3D_ERR.png")

    print("\n  3D ")

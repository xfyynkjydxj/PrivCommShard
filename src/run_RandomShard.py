import numpy as np
import matplotlib.pyplot as plt
from community_detector import CommunityDetector
from graph_builder import TransactionGraphBuilder
from metrics import compute_ctx
from shard_scheduler import ShardScheduler


def run_experiment():

    builder = TransactionGraphBuilder()
    detector = CommunityDetector()

    graphs = builder.generate_all_windows()

    Q_list = []
    CTX_list = []

    for i, G in enumerate(graphs):

        print(f"\nProcessing Window {i+1}")

        # 1. Q
        partition, Q = detector.detect_communities(G)
        Q_list.append(Q)

        # 2.  shard mapping
        accounts = list(G.nodes())
        scheduler = ShardScheduler(shard_num=6)

        # CommunityShard
        shard_map = {}
        for node, community in partition.items():
            scheduler = ShardScheduler(6)
            shard_map = scheduler.random_shard(accounts)


        # 3.  CTX
        df = builder.df.iloc[i*10000:(i+1)*10000]
        ctx = compute_ctx(df, shard_map)

        CTX_list.append(ctx)

        print(f"Q = {Q:.4f}, CTX = {ctx:.4f}")

    print("\n==== FINAL RESULT ====")
    print("Avg Q:", np.mean(Q_list))
    print("Avg CTX:", np.mean(CTX_list))

    plt.figure(figsize=(8, 5))

    plt.plot(Q_list)

    plt.xlabel("Window Index")
    plt.ylabel("Modularity Q")
    plt.title("Community Modularity over Sliding Windows")

    plt.grid(True)

    plt.savefig("../results/figures/Q_curve.png")

    plt.show()

    plt.figure(figsize=(8, 5))

    plt.plot(CTX_list)

    plt.xlabel("Window Index")
    plt.ylabel("CTX Ratio")
    plt.title("Cross-Shard Transaction Ratio")

    plt.grid(True)

    plt.savefig("../results/figures/CTX_curve.png")

    plt.show()


if __name__ == "__main__":
    run_experiment()

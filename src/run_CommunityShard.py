import pandas as pd
import networkx as nx
import community as community_louvain
import numpy as np

# CONFIG

INPUT_PATH = "../data/ethereum_clean.txt"

WINDOW_SIZE = 10000
STEP_SIZE = 5000

SHARD_NUM = 10

# BUILD TRANSACTION GRAPH

def build_window_graph(window_df):

    G = nx.Graph()

    for _, row in window_df.iterrows():

        sender = row["From"]
        receiver = row["To"]

        if pd.isna(sender) or pd.isna(receiver):
            continue

        if G.has_edge(sender, receiver):

            G[sender][receiver]["weight"] += 1

        else:

            G.add_edge(sender, receiver, weight=1)

    return G

# COMMUNITY DETECTION

def detect_communities(G):

    partition = community_louvain.best_partition(
        G,
        weight="weight"
    )

    Q = community_louvain.modularity(
        partition,
        G,
        weight="weight"
    )

    return partition, Q

# COMMUNITY SHARD MAPPING

def community_shard(partition, K):

    shard_map = {}

    for node, community_id in partition.items():

        #  shard
        shard_map[node] = community_id % K

    return shard_map

# COMPUTE CTX

def compute_ctx(window_df, shard_map):

    total_tx = 0
    cross_tx = 0

    for _, row in window_df.iterrows():

        sender = row["From"]
        receiver = row["To"]

        if pd.isna(sender) or pd.isna(receiver):
            continue

        #  shard_map
        if sender not in shard_map:
            continue

        if receiver not in shard_map:
            continue

        total_tx += 1

        if shard_map[sender] != shard_map[receiver]:

            cross_tx += 1

    if total_tx == 0:
        return 0

    return cross_tx / total_tx

# MAIN EXPERIMENT

def run_experiment():

    print("===================================")
    print("Loading Ethereum Dataset...")
    print("===================================")

    df = pd.read_csv(INPUT_PATH, sep="\t")

    print("Total Transactions:", len(df))

    total = len(df)

    Q_list = []
    CTX_list = []

    window_count = 0

    # SLIDING WINDOW

    for start in range(0, total - WINDOW_SIZE, STEP_SIZE):

        end = start + WINDOW_SIZE

        window_df = df.iloc[start:end]

        print(f"\nProcessing Window {window_count + 1}")

        # -------------------------------------------------
        # Build Transaction Graph
        # -------------------------------------------------

        G = build_window_graph(window_df)

        # -------------------------------------------------
        # Community Detection
        # -------------------------------------------------

        partition, Q = detect_communities(G)

        Q_list.append(Q)

        # -------------------------------------------------
        # CommunityShard Mapping
        # -------------------------------------------------

        shard_map = community_shard(
            partition,
            SHARD_NUM
        )

        # -------------------------------------------------
        # Compute CTX
        # -------------------------------------------------

        ctx = compute_ctx(
            window_df,
            shard_map
        )

        CTX_list.append(ctx)

        print(f"Q = {Q:.4f}")
        print(f"CTX = {ctx:.4f}")

        window_count += 1

    # FINAL RESULT

    print("\n===================================")
    print("FINAL RESULT")
    print("===================================")

    print("Shard Num:", SHARD_NUM)

    print("Window Size:", WINDOW_SIZE)

    print("Step Size:", STEP_SIZE)

    print("\nAvg Q:", np.mean(Q_list))

    print("Avg CTX:", np.mean(CTX_list))

# ENTRY

if __name__ == "__main__":

    run_experiment()

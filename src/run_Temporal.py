import pandas as pd
import networkx as nx
import random
import hashlib
import community as community_louvain
import numpy as np

# CONFIG

INPUT_PATH = "../data/ethereum_clean.txt"

WINDOW_SIZE = 10000
STEP_SIZE = 5000

SHARD_NUM = 10

BETA = 0.7

# MODE:
# random
# monoxide
# community
# temporal

MODE = "temporal"

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

# TEMPORAL GRAPH FUSION

def temporal_graph_fusion(current_G, previous_G, beta=0.7):

    fused_G = nx.Graph()

    for u, v, data in current_G.edges(data=True):

        current_weight = data["weight"]

        previous_weight = 0

        if previous_G is not None:

            if previous_G.has_edge(u, v):

                previous_weight = previous_G[u][v]["weight"]

        fused_weight = (
            beta * current_weight
            + (1 - beta) * previous_weight
        )

        fused_G.add_edge(u, v, weight=fused_weight)

    return fused_G

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

# RANDOM SHARD

def random_shard(accounts, K):

    mapping = {}

    for acc in accounts:

        mapping[acc] = random.randint(0, K - 1)

    return mapping

# MONOXIDE SHARD

def monoxide_shard(accounts, K):

    mapping = {}

    for acc in accounts:

        h = int(hashlib.md5(acc.encode()).hexdigest(), 16)

        mapping[acc] = h % K

    return mapping

# COMMUNITY SHARD

def community_shard(partition, K):

    mapping = {}

    for node, community_id in partition.items():

        mapping[node] = community_id % K

    return mapping

# COMPUTE CTX

def compute_ctx(window_df, shard_map):

    cross_tx = 0
    total_tx = 0

    for _, row in window_df.iterrows():

        sender = row["From"]
        receiver = row["To"]

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

    previous_G = None

    window_count = 0

    # SLIDING WINDOWS

    for start in range(0, total - WINDOW_SIZE, STEP_SIZE):

        end = start + WINDOW_SIZE

        window_df = df.iloc[start:end]

        print(f"\nProcessing Window {window_count + 1}")

        # -------------------------------------------------
        # Build Current Graph
        # -------------------------------------------------

        current_G = build_window_graph(window_df)

        # -------------------------------------------------
        # Temporal Fusion
        # -------------------------------------------------

        if MODE == "temporal":

            if previous_G is None:

                G = current_G

            else:

                G = temporal_graph_fusion(
                    current_G,
                    previous_G,
                    beta=BETA
                )

        else:

            G = current_G

        # -------------------------------------------------
        # Community Detection
        # -------------------------------------------------

        partition, Q = detect_communities(G)

        Q_list.append(Q)

        # -------------------------------------------------
        # Shard Mapping
        # -------------------------------------------------

        accounts = list(G.nodes())

        if MODE == "random":

            shard_map = random_shard(
                accounts,
                SHARD_NUM
            )

        elif MODE == "monoxide":

            shard_map = monoxide_shard(
                accounts,
                SHARD_NUM
            )

        elif MODE == "community":

            shard_map = community_shard(
                partition,
                SHARD_NUM
            )

        elif MODE == "temporal":

            shard_map = community_shard(
                partition,
                SHARD_NUM
            )

        else:

            raise ValueError("Invalid MODE")

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

        previous_G = current_G

        window_count += 1

    # FINAL RESULT

    print("\n===================================")
    print("FINAL RESULT")
    print("===================================")

    print("Mode:", MODE)

    print("Shard Num:", SHARD_NUM)

    print("Window Size:", WINDOW_SIZE)

    print("Step Size:", STEP_SIZE)

    print("Beta:", BETA)

    print("\nAvg Q:", np.mean(Q_list))

    print("Avg CTX:", np.mean(CTX_list))


# ENTRY

if __name__ == "__main__":

    run_experiment()

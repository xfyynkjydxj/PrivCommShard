import pandas as pd
import json
import random
import networkx as nx
from collections import defaultdict, Counter

CSV_PATH = "../data/ethereum_clean.txt"
SHARD_COUNT = 64  #  K
MAX_ITERATIONS = 100  # CLPA 


def load_eth_data(csv_path):
    """"""
    df = pd.read_csv(csv_path, sep="\t")
    df = df[df["isError"] == 0]
    zero_address = "0x0000000000000000000000000000000000000000"
    df = df[(df["From"] != zero_address) & (df["To"] != zero_address)]
    df = df.dropna(subset=["From", "To"])
    txs = list(zip(df["From"], df["To"]))
    print(f"  | {len(txs)}")
    return txs


def build_transaction_graph(txs):
    """"""
    G = nx.Graph()
    edge_weights = defaultdict(int)
    for f, t in txs:
        edge_weights[(f, t)] += 1
        edge_weights[(t, f)] += 1
    for (u, v), weight in edge_weights.items():
        G.add_edge(u, v, weight=weight)
    print(f"  | {G.number_of_nodes()}, {G.number_of_edges()}")
    return G


# ===================== CLPA  (1) =====================
def constrained_label_propagation(G, k):
    """
     Constrained Label Propagation Algorithm (CLPA)
    :param G: 
    :param k: 
    :return:  {account: community_id}
    """
    print(f"  CLPA  (k={k})...")
    nodes = list(G.nodes())
    n = len(nodes)
    max_size = n // k

    # Step 1: 
    label = {node: i for i, node in enumerate(nodes)}
    # ID -> 
    communities = defaultdict(list)
    for node, cid in label.items():
        communities[cid].append(node)

    for _ in range(MAX_ITERATIONS):
        updated = False
        # Step 2: 
        random.shuffle(nodes)

        for node in nodes:
            if node not in G:
                continue

            # Step 3: 
            neighbor_labels = defaultdict(int)
            for neighbor in G.neighbors(node):
                weight = G[node][neighbor]['weight']
                neighbor_labels[label[neighbor]] += weight

            if not neighbor_labels:
                continue

            # Step 4: 
            max_freq = max(neighbor_labels.values())
            candidates = [cid for cid, freq in neighbor_labels.items() if freq == max_freq]

            # Step 5:  - 
            current_cid = label[node]
            best_cid = current_cid
            best_size = len(communities[current_cid]) - 1

            for cid in candidates:
                candidate_size = len(communities[cid])
                if candidate_size < max_size and candidate_size > best_size:
                    best_cid = cid
                    best_size = candidate_size

            # Step 6: 
            if best_cid != current_cid:
                communities[current_cid].remove(node)
                if not communities[current_cid]:
                    del communities[current_cid]
                communities[best_cid].append(node)
                label[node] = best_cid
                updated = True

        if not updated:
            break

    print(f" CLPA  | {len(communities)}")
    return label


def map_communities_to_shards(community_label, k):
    """"""
    communities = defaultdict(list)
    for account, cid in community_label.items():
        communities[cid].append(account)
    sorted_communities = sorted(communities.values(), key=lambda x: len(x), reverse=True)

    shards = [[] for _ in range(k)]
    account_to_shard = {}

    for comm in sorted_communities:
        target_shard = min(range(k), key=lambda i: len(shards[i]))
        shards[target_shard].extend(comm)
        for account in comm:
            account_to_shard[account] = target_shard

    for i, shard in enumerate(shards):
        print(f" {i}  {len(shard)} ")

    return account_to_shard


# =====================  (CTX) =====================
def calculate_cross_shard_rate(txs, account_to_shard):
    """"""
    total_txs = len(txs)
    cross_txs = 0
    intra_txs = 0

    for f, t in txs:
        if f not in account_to_shard or t not in account_to_shard:
            cross_txs += 1
            continue
        if account_to_shard[f] == account_to_shard[t]:
            intra_txs += 1
        else:
            cross_txs += 1

    ctx_rate = cross_txs / total_txs
    print("\n" + "=" * 60)
    print(f" ShardCutter (Huang 2026) CTX ")
    print(f"{total_txs}")
    print(f"{intra_txs}")
    print(f"{cross_txs}")
    print(f" CTX{ctx_rate:.4f} = {ctx_rate * 100:.2f}%")
    print("=" * 60)
    return ctx_rate


if __name__ == "__main__":
    print("  ShardCutter (Huang 2026) \n")

    # 1. 
    txs = load_eth_data(CSV_PATH)

    # 2. 
    G = build_transaction_graph(txs)

    # 3.  CLPA 
    community_label = constrained_label_propagation(G, SHARD_COUNT)

    # 4. 
    account_to_shard = map_communities_to_shards(community_label, SHARD_COUNT)

    # 5.  CTX
    calculate_cross_shard_rate(txs, account_to_shard)

    # 6. 
    with open("shardcutter_clpa_mapping.json", "w", encoding="utf-8") as f:
        json.dump(account_to_shard, f, indent=2)

    print(f"\n   shardcutter_clpa_mapping.json")

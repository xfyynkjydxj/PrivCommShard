import csv
import random
import numpy as np
import networkx as nx
from collections import defaultdict, Counter

WINDOW_SIZE = 10000
MAX_WINDOWS = 10
SHARD_COUNT = 64     #  K
LPA_ITER = 10           # CLPA  T
BETA = 0.5              #   0.5
CSV_PATH = "../data/ethereum_clean.txt"

INIT_RANDOM_RATIO = 0.0   # 0=0.1=10%

# 1. 
def read_windows():
    windows = []
    current = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s = row["From"].strip()
            r = row["To"].strip()
            if s and r:
                current.append((s, r))
                if len(current) >= WINDOW_SIZE:
                    windows.append(current)
                    current = []
                    if len(windows) >= MAX_WINDOWS:
                        break
    print("Total Windows:", len(windows))
    return windows

# 2. 
def build_graph(txs):
    G = nx.Graph()
    for u, v in txs:
        if G.has_edge(u, v):
            G[u][v]["weight"] += 1
        else:
            G.add_edge(u, v, weight=1)
    return G

# 3. CLPAW_k = 
def clpa_shardcutter(G, initial_labels):
    labels = initial_labels.copy()
    nodes = list(G.nodes())

    for _ in range(LPA_ITER):
        random.shuffle(nodes)
        # ----------  W_k ----------
        comm_weight = defaultdict(float)
        for u, v in G.edges():
            c = labels[u]
            comm_weight[c] += G[u][v]["weight"]
        if not comm_weight:
            continue
        min_weight = min(comm_weight.values())
        if min_weight == 0:
            min_weight = 1e-9

        for node in nodes:
            neighbor_weights = defaultdict(float)
            total_weight = sum(G[node][nei]["weight"] for nei in G.neighbors(node))
            if total_weight == 0:
                continue

            # ---------- s(i,k) ----------
            for nei in G.neighbors(node):
                c = labels[nei]
                w = G[node][nei]["weight"]
                # 1 -  * W_k / min_h W_h
                penalty = 1.0 - BETA * (comm_weight[c] / min_weight)
                neighbor_weights[c] += (w / total_weight) * penalty

            if neighbor_weights:
                labels[node] = max(neighbor_weights, key=neighbor_weights.get)
    return labels

# 4.   
def assign_communities_to_shards(comm_nodes, k):
    shard_map = {}
    load = [0] * k
    comms = sorted(comm_nodes.values(), key=len, reverse=True)
    for nodes in comms:
        sid = load.index(min(load))
        for n in nodes:
            shard_map[n] = sid
        load[sid] += len(nodes)
    return shard_map

# 5. CTXSSILV
def calc_ctx(txs, shard):
    cross, total = 0, 0
    for u, v in txs:
        if u in shard and v in shard:
            total += 1
            if shard[u] != shard[v]:
                cross += 1
    return cross / total if total else 0.0

def calc_ssi(prev, curr):
    common = [k for k in prev if k in curr]
    if not common:
        return 1.0
    mig = sum(1 for x in common if prev[x] != curr[x])
    return 1.0 - mig / len(common)

def calc_lv(m):
    c = Counter(m.values())
    return np.var(list(c.values())) if c else 0.0

# MAIN
def main():
    windows = read_windows()
    ctx_list, ssi_list, lv_list = [], [], []
    prev_shard = {}

    print("\n===== ShardCutter (Exact Original, W_k=) =====")
    for i, txs in enumerate(windows):
        G = build_graph(txs)

        #  + 
        init_labels = {}
        for n in G.nodes():
            if random.random() < INIT_RANDOM_RATIO:
                init_labels[n] = random.randint(0, SHARD_COUNT-1)
            else:
                init_labels[n] = prev_shard.get(n, random.randint(0, SHARD_COUNT-1))

        labels = clpa_shardcutter(G, init_labels)

        comm = defaultdict(list)
        for n, c in labels.items():
            comm[c].append(n)

        curr_shard = assign_communities_to_shards(comm, SHARD_COUNT)

        ctx = calc_ctx(txs, curr_shard)
        lv = calc_lv(curr_shard)
        ctx_list.append(ctx)
        lv_list.append(lv)

        if i > 0:
            ssi = calc_ssi(prev_shard, curr_shard)
            ssi_list.append(ssi)
            print(f"Win{i+1:2d} | CTX={ctx:.4f} | SSI={ssi:.4f} | LV={lv:.1f}")
        else:
            print(f"Win{i+1:2d} | CTX={ctx:.4f} | LV={lv:.1f}")

        prev_shard = curr_shard

    print("\n================ FINAL ================")
    print(f"Avg CTX = {np.mean(ctx_list):.4f}")
    print(f"Avg SSI = {np.mean(ssi_list):.4f}")
    print(f"Avg LV  = {np.mean(lv_list):.2f}")

if __name__ == "__main__":
    main()

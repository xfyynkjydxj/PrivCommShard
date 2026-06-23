import csv
import random
import numpy as np
import networkx as nx

from collections import defaultdict, Counter
from community import community_louvain

# CONFIG  
WINDOW_SIZE = 10000
MAX_WINDOWS = 10
SHARD_COUNT = 64

EPS = 0.5
LAMBDA = 0.8       #   ALPHA

CSV_PATH = "../data/ethereum_clean.txt"

# READ DATA
def read_windows():
    windows = []
    current = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sender = row["From"].strip()
            receiver = row["To"].strip()
            if sender and receiver:
                current.append((sender, receiver))
                if len(current) >= WINDOW_SIZE:
                    windows.append(current)
                    current = []
                    if len(windows) >= MAX_WINDOWS:
                        break
    print("Total Windows =", len(windows))
    return windows

# BUILD GRAPH
def build_graph(txs):
    G = nx.Graph()
    for u, v in txs:
        if G.has_edge(u, v):
            G[u][v]["weight"] += 1
        else:
            G.add_edge(u, v, weight=1)
    return G

#   + 
def dp_perturb(G, eps):
    dpG = nx.Graph()
    dpG.add_nodes_from(G.nodes())

    for u, v, d in G.edges(data=True):
        w = d["weight"]
        p = np.exp(-eps * w) / (1 + np.exp(-eps * w))
        if random.random() > p:
            dpG.add_edge(u, v, weight=w)
    return dpG

def temporal_fusion(curr, prev):
    if prev is None:
        return curr
    fused = curr.copy()
    for u, v, d in prev.edges(data=True):
        hist_w = d["weight"] * LAMBDA
        if fused.has_edge(u, v):
            fused[u][v]["weight"] += hist_w
        else:
            fused.add_edge(u, v, weight=hist_w)
    return fused

# COMMUNITY DETECTION
def detect_community(G):
    return community_louvain.best_partition(
        G,
        weight="weight",
        resolution=1.0
    )

# COMMUNITY GROUP
def build_community_group(partition):
    groups = defaultdict(set)
    for node, cid in partition.items():
        groups[cid].add(node)
    return groups

# JACCARD MATCHING
def match_community(curr_nodes, prev_groups):
    best_comm = None
    best_score = 0
    for pid, prev_nodes in prev_groups.items():
        inter = len(curr_nodes & prev_nodes)
        union = len(curr_nodes | prev_nodes)
        if union == 0:
            continue
        score = inter / union
        if score > best_score:
            best_score = score
            best_comm = pid
    return best_comm

# SHARD ASSIGNMENT
def assign_shard(curr_groups, prev_groups, prev_shard, shard_num):
    shard_map = {}
    load = [0] * shard_num
    for s in prev_shard.values():
        load[s] += 1

    for cid, nodes in curr_groups.items():
        matched = match_community(nodes, prev_groups)
        votes = Counter()
        if matched is not None:
            for node in nodes:
                if node in prev_shard:
                    votes[prev_shard[node]] += 1
        if votes:
            shard_id = votes.most_common(1)[0][0]
        else:
            shard_id = load.index(min(load))
        for node in nodes:
            shard_map[node] = shard_id
        load[shard_id] += len(nodes)
    return shard_map

# METRICS
def calc_ctx(txs, shard):
    cross = total = 0
    for u, v in txs:
        if u in shard and v in shard:
            total += 1
            if shard[u] != shard[v]:
                cross += 1
    return cross / total if total else 0

def calc_ssi(prev_shard, curr_shard):
    common = set(prev_shard) & set(curr_shard)
    if not common:
        return 1.0
    mig = sum(1 for n in common if prev_shard[n] != curr_shard[n])
    return 1 - mig / len(common)

def calc_err(orig, dp):
    o = set(orig.edges())
    d = set(dp.edges())
    return len(o & d) / len(o) if o else 0

# MAIN
def main():
    windows = read_windows()
    ctx_list, ssi_list, err_list = [], [], []
    prev_graph = None
    prev_groups = {}
    prev_shard = {}

    print("\n===== PrivCommShard Final =====\n")
    for idx, txs in enumerate(windows):
        print(f"Window {idx+1}")
        G = build_graph(txs)
        G_dp = dp_perturb(G, EPS)
        err = calc_err(G, G_dp)
        G_fused = temporal_fusion(G_dp, prev_graph)
        partition = detect_community(G_fused)
        curr_groups = build_community_group(partition)
        curr_shard = assign_shard(curr_groups, prev_groups, prev_shard, SHARD_COUNT)
        ctx = calc_ctx(txs, curr_shard)
        ctx_list.append(ctx)
        err_list.append(err)

        if idx > 0:
            ssi = calc_ssi(prev_shard, curr_shard)
            ssi_list.append(ssi)
            print(f"CTX={ctx:.4f} | SSI={ssi:.4f} | ERR={err:.4f}\n")
        else:
            print(f"CTX={ctx:.4f} | ERR={err:.4f}\n")

        prev_graph = G_fused
        prev_groups = curr_groups
        prev_shard = curr_shard

    print("================ FINAL ================")
    print(f"Avg CTX = {np.mean(ctx_list):.4f}")
    print(f"Avg SSI = {np.mean(ssi_list):.4f}")
    print(f"Avg ERR = {np.mean(err_list):.4f}")

if __name__ == "__main__":
    main()

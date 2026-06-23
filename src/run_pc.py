import pandas as pd
import json
import random
import numpy as np
import networkx as nx
from community import community_louvain
from collections import defaultdict

CSV_PATH = "../data/ethereum_clean.txt"
SHARD_COUNT = 64

EPS = 0.5
MAX_LOAD_RATE = 1.2


def load_full_eth_data(csv_path):
    df = pd.read_csv(csv_path, sep="\t")
    zero_addr = "0x0000000000000000000000000000000000000000"
    df = df[(df["isError"] == 0) & (df["Value"] > 0)]
    df = df[(df["From"] != zero_addr) & (df["To"] != zero_addr)]
    txs = list(zip(df["From"], df["To"]))
    print(f"  | {len(txs)}")
    return txs


def build_global_graph(txs):
    G = nx.Graph()
    freq = defaultdict(int)
    for f, t in txs:
        freq[(f, t)] += 1
        freq[(t, f)] += 1
    for (u, v), w in freq.items():
        G.add_edge(u, v, weight=w)
    return G


def dp_perturb(G, eps=EPS):
    Gp = nx.Graph()
    for u, v, d in G.edges(data=True):
        w = d["weight"]

        x = eps * w
        if x > 20:  # 20 exp(x) 
            p = 1.0
        else:
            p = np.exp(x) / (1 + np.exp(x))

        if random.random() < p:
            Gp.add_edge(u, v, weight=w)
    return Gp


def constrained_louvain(G):
    return community_louvain.best_partition(G, weight="weight")


def shard_mapping(comm):
    groups = defaultdict(list)
    for node, cid in comm.items():
        groups[cid].append(node)

    max_load = int(len(comm) / SHARD_COUNT * MAX_LOAD_RATE)
    shard_load = {i: 0 for i in range(SHARD_COUNT)}
    shard_map = {}

    for cid, nodes in groups.items():
        sid = min(shard_load, key=shard_load.get)
        if shard_load[sid] + len(nodes) > max_load:
            sid = min([k for k, v in shard_load.items() if v < max_load], default=0)
        for node in nodes:
            shard_map[node] = sid
        shard_load[sid] += len(nodes)
    return shard_map, shard_load


# =====================  CTX  =====================
def calculate_full_ctx(txs, shard_map):
    total = len(txs)
    cross = 0
    intra = 0

    for f, t in txs:
        if f not in shard_map or t not in shard_map:
            cross += 1
            continue
        if shard_map[f] == shard_map[t]:
            intra += 1
        else:
            cross += 1

    ctx = cross / total
    print("\n" + "=" * 60)
    print(f"  CTX ")
    print(f"{total}")
    print(f"{intra}")
    print(f"{cross}")
    print(f" CTX{ctx:.4f} = {ctx * 100:.2f}%")
    print("=" * 60)
    return ctx


if __name__ == "__main__":
    print(" PrivCommShard-DP Enhanced   CTX \n")

    # 1. 
    txs = load_full_eth_data(CSV_PATH)

    # 2. 
    G = build_global_graph(txs)

    # 3. 
    G_dp = dp_perturb(G)

    # 4. 
    comm = constrained_louvain(G_dp)

    # 5. 
    shard_map, shard_load = shard_mapping(comm)

    # 6.  CTX
    calculate_full_ctx(txs, shard_map)

    # 7. 
    with open("privcomm_shard_full_ctx.json", "w", encoding="utf-8") as f:
        json.dump(shard_map, f, indent=2)

    print(f"\n   privcomm_shard_full_ctx.json")

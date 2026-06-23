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
CSV_PATH = "../data/ethereum_clean.txt"
OUTPUT_CSV = "../data/experiment_results64.csv"

EPS_LIST = np.linspace(0.1, 5, 50)
LAMBDA_LIST = np.linspace(0.1, 1.00, 10)

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

# dp
def dp_perturb(G, eps):
    dpG = nx.Graph()
    dpG.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        w = d["weight"]
        p = np.exp(-eps * w) / (1 + np.exp(-eps * w))
        if random.random() > p:
            dpG.add_edge(u, v, weight=w)
    return dpG

# tem
def temporal_fusion(curr, prev, lam):
    if prev is None:
        return curr
    fused = curr.copy()
    for u, v, d in prev.edges(data=True):
        hist_w = d["weight"] * lam
        if fused.has_edge(u, v):
            fused[u][v]["weight"] += hist_w
        else:
            fused.add_edge(u, v, weight=hist_w)
    return fused

# com
def detect_community(G):
    return community_louvain.best_partition(G, weight="weight", resolution=1.0)

def build_community_group(partition):
    groups = defaultdict(set)
    for node, cid in partition.items():
        groups[cid].add(node)
    return groups

def match_community(curr_nodes, prev_groups):
    best_comm, best_score = None, 0
    for pid, prev_nodes in prev_groups.items():
        inter = len(curr_nodes & prev_nodes)
        union = len(curr_nodes | prev_nodes)
        if union == 0: continue
        score = inter / union
        if score > best_score:
            best_score, best_comm = score, pid
    return best_comm

def assign_shard(curr_groups, prev_groups, prev_shard, shard_num):
    shard_map = {}
    load = [0] * shard_num
    for s in prev_shard.values(): load[s] += 1
    for cid, nodes in curr_groups.items():
        matched = match_community(nodes, prev_groups)
        votes = Counter()
        if matched is not None:
            for node in nodes:
                if node in prev_shard: votes[prev_shard[node]] += 1
        if votes:
            shard_id = votes.most_common(1)[0][0]
        else:
            shard_id = load.index(min(load))
        for node in nodes: shard_map[node] = shard_id
        load[shard_id] += len(nodes)
    return shard_map

# METRICS
def calc_ctx(txs, shard):
    cross = total = 0
    for u, v in txs:
        if u in shard and v in shard:
            total += 1
            if shard[u] != shard[v]: cross += 1
    return cross / total if total else 0

def calc_ssi(prev_shard, curr_shard):
    common = set(prev_shard) & set(curr_shard)
    if not common: return 1.0
    mig = sum(1 for n in common if prev_shard[n] != curr_shard[n])
    return 1 - mig / len(common)

def calc_err(orig, dp):
    o = set(orig.edges())
    d = set(dp.edges())
    return len(o & d) / len(o) if o else 0

# 1time
def run_once(windows, eps, lam):
    ctx_list, ssi_list, err_list = [], [], []
    prev_graph = None
    prev_groups = {}
    prev_shard = {}

    for idx, txs in enumerate(windows):
        G = build_graph(txs)
        G_dp = dp_perturb(G, eps)
        err = calc_err(G, G_dp)
        G_fused = temporal_fusion(G_dp, prev_graph, lam)
        partition = detect_community(G_fused)
        curr_groups = build_community_group(partition)
        curr_shard = assign_shard(curr_groups, prev_groups, prev_shard, SHARD_COUNT)
        ctx = calc_ctx(txs, curr_shard)
        ctx_list.append(ctx)
        err_list.append(err)
        if idx > 0:
            ssi = calc_ssi(prev_shard, curr_shard)
            ssi_list.append(ssi)
        prev_graph, prev_groups, prev_shard = G_fused, curr_groups, curr_shard

    return np.mean(ctx_list), np.mean(ssi_list), np.mean(err_list)

#  CSV
def run_experiment_and_save_csv():
    windows = read_windows()

    # openCSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # head
        writer.writerow(["epsilon", "lambda", "CTX", "SSI", "ERR(RA)"])

        print("\n round    open CSV...")
        for i, eps in enumerate(EPS_LIST):
            for j, lam in enumerate(LAMBDA_LIST):
                ctx, ssi, err = run_once(windows, eps, lam)
                # 1line
                writer.writerow([
                    round(eps, 4),
                    round(lam, 4),
                    round(ctx, 6),
                    round(ssi, 6),
                    round(err, 6)
                ])
                print(f"={eps:.2f}, ={lam:.2f} | CTX={ctx:.4f}, SSI={ssi:.4f}, ERR={err:.4f}")

    print(f"\n no err {OUTPUT_CSV}")

# MAIN
if __name__ == "__main__":
    run_experiment_and_save_csv()

import csv
import random
import hashlib
import numpy as np
from collections import defaultdict

WINDOW_SIZE = 10000
MAX_WINDOWS = 10
SHARD_COUNT = 32
CSV_PATH = "../data/ethereum_clean.txt"

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
    return windows

# Monoxide 
def address_to_shard(addr, shard_num):
    h = hashlib.sha256(addr.encode()).hexdigest()
    return int(h, 16) % shard_num

def get_shard_map(txs, shard_num):
    shard = {}
    for s, r in txs:
        if s not in shard:
            shard[s] = address_to_shard(s, shard_num)
        if r not in shard:
            shard[r] = address_to_shard(r, shard_num)
    return shard

#  CTX
def calc_ctx(txs, shard_map):
    cross = 0
    total = 0
    for s, r in txs:
        if s not in shard_map or r not in shard_map:
            continue
        total += 1
        if shard_map[s] != shard_map[r]:
            cross += 1
    return cross / total if total else 0

#  SSI
def calc_ssi(prev_shard, curr_shard):
    common = set(prev_shard.keys()) & set(curr_shard.keys())
    if not common:
        return 1.0
    changed = 0
    for node in common:
        if prev_shard[node] != curr_shard[node]:
            changed += 1
    return 1.0 - (changed / len(common))

def main():
    windows = read_windows()
    ctx_list = []
    ssi_list = []
    prev_shard = None

    print("\n===== Monoxide (NSDI 19) =====")
    for i, txs in enumerate(windows):
        curr_shard = get_shard_map(txs, SHARD_COUNT)
        ctx = calc_ctx(txs, curr_shard)
        ctx_list.append(ctx)

        if i > 0 and prev_shard is not None:
            ssi = calc_ssi(prev_shard, curr_shard)
            ssi_list.append(ssi)
            print(f"Window {i+1} | CTX={ctx:.4f} | SSI={ssi:.4f}")
        else:
            print(f"Window {i+1} | CTX={ctx:.4f}")

        prev_shard = curr_shard

    print("\n================ FINAL ================")
    print(f"Monoxide Avg CTX = {np.mean(ctx_list):.4f}")
    if ssi_list:
        print(f"Monoxide Avg SSI = {np.mean(ssi_list):.4f}")

if __name__ == "__main__":
    main()

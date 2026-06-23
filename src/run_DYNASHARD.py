import pandas as pd
import hashlib
import random
from collections import defaultdict

# ===================== DYNASHARD  =====================
K = 2
HOT_THRESHOLD = 1.5
random.seed(42)


def load_data(file_path):
    df = pd.read_csv(file_path, sep="\t")
    df = df[(df['isError'] == 0) & (df['Value'] > 0)].dropna(subset=['From', 'To'])
    zero_addr = '0x0000000000000000000000000000000000000000'
    df = df[(df['From'] != zero_addr) & (df['To'] != zero_addr)]
    return df


def get_accounts(df):
    return list(set(df.From).union(set(df.To)))


def hash_shard(accounts, K):
    m = {}
    for acc in accounts:
        h = int(hashlib.sha256(acc.encode()).hexdigest(), 16)
        m[acc] = h % K
    return m


def dynamic_adjust(df, init_map):
    shards = defaultdict(list)
    for acc, sid in init_map.items():
        shards[sid].append(acc)

    load = defaultdict(int)
    for _, row in df.iterrows():
        s = row['From']
        if s in init_map:
            load[init_map[s]] += 1

    if not load:
        return init_map

    avg = sum(load.values()) / len(load)
    hot = [k for k, v in load.items() if v > HOT_THRESHOLD * avg]
    if hot:
        h = hot[0]
        members = shards[h]
        if len(members) > 1:
            new_sid = max(shards.keys()) + 1
            half = len(members) // 2
            for acc in members[half:]:
                init_map[acc] = new_sid
    return init_map


def calculate_ctx(df, shard_map):
    total, cross = 0, 0
    for _, row in df.iterrows():
        s, r = row['From'], row['To']
        if s in shard_map and r in shard_map:
            total += 1
            if shard_map[s] != shard_map[r]:
                cross += 1
    return cross / total if total else 0


if __name__ == "__main__":
    df = load_data("../data/ethereum_clean.txt")
    accounts = get_accounts(df)
    init_map = hash_shard(accounts, K)
    final_map = dynamic_adjust(df, init_map)
    ctx = calculate_ctx(df, final_map)

    print("=" * 50)
    print("DYNASHARD ")
    print(f"CTX = {ctx:.4f}")
    print("=" * 50)

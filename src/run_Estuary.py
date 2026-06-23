import pandas as pd
import random
import json
from collections import defaultdict

CSV_PATH = "../data/ethereum_clean.txt"
SHARD_NUM = 10
STATE_SPLIT_RATIO = 0.5#  50%
random.seed(42)

def load_eth_data(csv_path):
    df = pd.read_csv(csv_path, sep="\t")
    df = df[(df["isError"] == 0) & (df["Value"] > 0)]
    zero_addr = "0x0000000000000000000000000000000000000000"
    df = df[(df["From"] != zero_addr) & (df["To"] != zero_addr)]
    return df

# ===================== Estuary  =====================
class AccountState:
    def __init__(self, addr, shard_count):
        self.addr = addr
        #  [0,1]
        self.belonging_coeff = {s: random.random() for s in range(shard_count)}
        self.primary_shard = max(self.belonging_coeff, key=self.belonging_coeff.get)
        #  50% 50%
        self.state_units = self._split_state(shard_count)

    def _split_state(self, shard_count):
        state = defaultdict(float)
        state[self.primary_shard] = STATE_SPLIT_RATIO
        rest = (1.0 - STATE_SPLIT_RATIO) / (shard_count - 1)
        for s in range(shard_count):
            if s != self.primary_shard:
                state[s] = rest
        return state

def community_overlap_propagation(df, account_map, shard_count):
    """/"""
    interact_graph = defaultdict(int)
    for _, row in df.iterrows():
        fr, to = row["From"], row["To"]
        if fr == to:
            continue
        interact_graph[(fr, to)] += 1
        interact_graph[(to, fr)] += 1

    interact_threshold = 5
    for (addr1, addr2), cnt in interact_graph.items():
        if cnt > interact_threshold:
            acc1 = account_map[addr1]
            acc2 = account_map[addr2]
            target_shard = acc1.primary_shard
            acc2.belonging_coeff[target_shard] += 0.25
            acc2.primary_shard = max(acc2.belonging_coeff, key=acc2.belonging_coeff.get)

            target_shard2 = acc2.primary_shard
            acc1.belonging_coeff[target_shard2] += 0.25
            acc1.primary_shard = max(acc1.belonging_coeff, key=acc1.belonging_coeff.get)

    return account_map

# =====================  CTX =====================
def calculate_real_ctx(account_map, total_tx=10000):
    """
    Estuary 
     >   (cross)
      (intra)
    """
    intra_cnt = 0
    cross_cnt = 0
    addr_list = list(account_map.keys())

    for _ in range(total_tx):
        addr = random.choice(addr_list)
        acc = account_map[addr]
        #  0~1
        tx_amount = random.uniform(0.05, 1.0)
        primary_state = acc.state_units[acc.primary_shard]

        if primary_state >= tx_amount:
            intra_cnt += 1
        else:
            cross_cnt += 1

    return cross_cnt / total_tx if total_tx > 0 else 0.0

def save_account_info(account_map, save_path="estuary_accounts.json"):
    dump_data = {}
    for addr, acc in account_map.items():
        dump_data[addr] = {
            "primary_shard": acc.primary_shard,
            "state_units": dict(acc.state_units)
        }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(dump_data, f, indent=2)

if __name__ == "__main__":
    print("=" * 60)
    print("        Estuary  + CTX CSV")
    print("=" * 60)

    # 1. 
    print(f"[1] : {CSV_PATH}")
    df = load_eth_data(CSV_PATH)
    all_addrs = set(df["From"]).union(set(df["To"]))
    print(f": {len(all_addrs)}")

    # 2.  & 
    print("[2]  + ")
    account_dict = {}
    for addr in all_addrs:
        account_dict[addr] = AccountState(addr, SHARD_NUM)

    # 3. 
    print("[3] ")
    account_dict = community_overlap_propagation(df, account_dict, SHARD_NUM)

    # 4.  CTX
    print("[4]  Estuary  CTX")
    ctx_value = calculate_real_ctx(account_dict, total_tx=10000)
    print(f"\n  CTX = {ctx_value:.4f}")

    # 5. 
    save_account_info(account_dict)
    print(" : estuary_accounts.json")
    print("=" * 60)

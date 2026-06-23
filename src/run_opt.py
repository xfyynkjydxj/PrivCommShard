import csv
import json
from collections import defaultdict

SHARD_COUNT = 64
LAMBDA = 0.01
CSV_PATH = "../data/ethereum_clean.txt"


def read_eth_transactions(csv_path):
    txs = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter="\t")
        print("CSV:", reader.fieldnames)

        for row in reader:
            #   From  To
            frm = row.get("From")
            to = row.get("To")

            if frm and to and len(frm) > 0 and len(to) > 0:
                txs.append({
                    "txid": len(txs),
                    "from": frm,
                    "to": to
                })

    print(f" {len(txs)} ")
    return txs


# ==================== OptChain  ====================
def build_transaction_dependency(txs):
    addr_last_tx = dict()
    tx_prev = dict()

    for tx in txs:
        txid = tx["txid"]
        frm = tx["from"]
        tx_prev[txid] = addr_last_tx.get(frm, None)
        addr_last_tx[tx["to"]] = txid

    return tx_prev


# ==================== OptChain  ====================
def optchain_online_sharding(txs, tx_prev, shard_num):
    shard_load = defaultdict(int)
    tx_to_shard = dict()

    for tx in txs:
        txid = tx["txid"]
        prev_txid = tx_prev[txid]

        # T2S 
        t2s_score = {}
        if prev_txid is not None and prev_txid in tx_to_shard:
            best_shard = tx_to_shard[prev_txid]
            for s in range(shard_num):
                t2s_score[s] = 1.0 if s == best_shard else 0.1
        else:
            for s in range(shard_num):
                t2s_score[s] = 1.0 / shard_num

        # L2S 
        l2s_score = {s: shard_load[s] for s in range(shard_num)}

        final_score = {s: t2s_score[s] - LAMBDA * l2s_score[s] for s in range(shard_num)}
        selected_shard = max(final_score, key=final_score.get)

        tx_to_shard[txid] = selected_shard
        shard_load[selected_shard] += 1

    print(f" OptChain  | {dict(shard_load)}")
    return tx_to_shard


# ====================  CTX ====================
def calculate_cross_shard_rate(txs, tx_to_shard, tx_prev):
    if len(txs) == 0:
        print(" CTX=0")
        return 0.0

    cross_shard_tx = 0
    total = len(txs)

    for tx in txs:
        txid = tx["txid"]
        current_shard = tx_to_shard[txid]
        prev_txid = tx_prev.get(txid, None)

        if prev_txid is None:
            continue

        if tx_to_shard.get(prev_txid, current_shard) != current_shard:
            cross_shard_tx += 1

    ctx = cross_shard_tx / total
    print("\n" + "=" * 60)
    print("           OptChain  CTX ")
    print(f"{total}")
    print(f"{cross_shard_tx}")
    print(f" CTX = {ctx:.4f} = {ctx * 100:.2f}%")
    print("=" * 60)
    return ctx


if __name__ == "__main__":
    tx_list = read_eth_transactions(CSV_PATH)
    if len(tx_list) == 0:
        print(" CSV")
        exit()

    tx_prev = build_transaction_dependency(tx_list)
    tx_shard_map = optchain_online_sharding(tx_list, tx_prev, SHARD_COUNT)
    ctx = calculate_cross_shard_rate(tx_list, tx_shard_map, tx_prev)

    with open("optchain_tx_shard_map.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in tx_shard_map.items()}, f, indent=2)

    print(" optchain_tx_shard_map.json")

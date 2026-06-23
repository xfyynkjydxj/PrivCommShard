import random
import time
import threading
from queue import Queue
from collections import defaultdict

TOTAL_TX = 10000
SHARD_COUNT = 4
NODE_PER_SHARD = 4        # PBFT  4 
CTX = 0.75                # ShardCutter=0.75=0.55
BLOCK_SIZE = 500

# ms 
PBFT_VOTE_MS = 4          # PBFT 
BLOCK_PACK_MS = 2
RELAY_NETWORK_MS = 10

tx_results = []
lock = threading.Lock()

# ===================== 1.  CTX  =====================
def generate_txs(ctx, total):
    txs = []
    for i in range(total):
        from_shard = random.randint(0, SHARD_COUNT-1)
        to_shard = from_shard if random.random() > ctx else random.choice(
            [s for s in range(SHARD_COUNT) if s != from_shard]
        )
        txs.append({
            "txid": i,
            "from_shard": from_shard,
            "to_shard": to_shard,
            "type": "INTRA" if from_shard == to_shard else "CROSS"
        })
    return txs

# ===================== 2. PBFT  =====================
def pbft_consensus(shard_id):
    nodes = []
    result = {"commit": 0}

    def node_task():
        time.sleep(PBFT_VOTE_MS / 1000)
        with lock:
            result["commit"] += 1

    for _ in range(NODE_PER_SHARD):
        t = threading.Thread(target=node_task)
        nodes.append(t)
        t.start()
    for t in nodes:
        t.join()
    return True

# ===================== 3.  =====================
def pack_block(txs_batch):
    time.sleep(BLOCK_PACK_MS / 1000)
    return {"tx_count": len(txs_batch), "timestamp": time.time()}

# ===================== 4.  =====================
def process_tx(tx):
    start = time.time()

    if tx["type"] == "INTRA":
        # PBFT    
        pbft_consensus(tx["from_shard"])
        pack_block([tx])

    else:
        # PBFT  Relay  PBFT
        pbft_consensus(tx["from_shard"])
        time.sleep(RELAY_NETWORK_MS / 1000)
        pbft_consensus(tx["to_shard"])
        pack_block([tx])

    latency = (time.time() - start) * 1000  #  ms

    with lock:
        tx_results.append((tx["txid"], latency))

# ===================== 5.  =====================
def run_simulation(txs):
    threads = []
    for tx in txs:
        t = threading.Thread(target=process_tx, args=(tx,))
        threads.append(t)
        t.start()
        if len(threads) >= 32:
            for t in threads:
                t.join()
            threads = []
    for t in threads:
        t.join()

# ===================== 6.  TPS /  =====================
def calculate_metrics(total_tx, start_time):
    elapsed = time.time() - start_time
    tps = total_tx / elapsed
    latencies = [l for _, l in tx_results]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    cross = len([t for t in txs if t["type"] == "CROSS"])
    intra = total_tx - cross
    return tps, avg_lat, elapsed, cross, intra

if __name__ == "__main__":
    print("=" * 80)
    print("        +  + ")
    print(f"       {SHARD_COUNT} | /{NODE_PER_SHARD} | CTX{CTX:.2%}")
    print("=" * 80)

    # 1. 
    txs = generate_txs(CTX, TOTAL_TX)

    # 2. 
    start_time = time.time()
    run_simulation(txs)
    tps, avg_lat, elapsed, cross, intra = calculate_metrics(TOTAL_TX, start_time)

    # 3. 
    print(f"\n {BLOCK_SIZE}")
    print(f" {intra} | {cross}")
    print(f" {elapsed:.3f} s")
    print(f" {avg_lat:.2f} ms")
    print(f" TPS{tps:.2f} tx/s")
    print("=" * 80)

import asyncio
import random
import time
import json
from collections import defaultdict

CTX = 0.5794
TOTAL_TX = 10000
SHARD_COUNT = 64             #  2/4/6/8/10
NODES_PER_SHARD = 4
BLOCK_SIZE = 500

class Block:
    def __init__(self, shard_id, height, tx_list, tx_type):
        self.shard_id = shard_id
        self.height = height
        self.tx_list = tx_list
        self.tx_count = len(tx_list)
        self.tx_type = tx_type
        self.tx_latencies = []  #   (ms)

    def add_latency(self, lat_ms):
        self.tx_latencies.append(lat_ms)

    @property
    def avg_latency(self):
        if not self.tx_latencies:
            return 0.0
        return sum(self.tx_latencies) / len(self.tx_latencies)

    def to_dict(self):
        return {
            "shard": self.shard_id,
            "height": self.height,
            "tx_count": self.tx_count,
            "tx_type": self.tx_type,
            "avg_latency_ms": round(self.avg_latency, 2)
        }

# ==================== PBFT  ====================
class Node:
    def __init__(self, node_id, shard_id):
        self.node_id = node_id
        self.shard_id = shard_id
        self.inbox = asyncio.Queue()
        self.prepared = set()
        self.committed = set()
        self.task = None

    async def broadcast(self, msg_type, tx_id, all_nodes):
        for node in all_nodes:
            if node.shard_id == self.shard_id:
                await node.inbox.put((msg_type, tx_id))

    async def run(self, all_nodes):
        while True:
            msg = await self.inbox.get()
            if msg is None:
                break
            msg_type, tx_id = msg

            if msg_type == "preprepare":
                self.prepared.add(tx_id)
                await self.broadcast("prepare", tx_id, all_nodes)

            elif msg_type == "prepare":
                self.prepared.add(tx_id)
                await self.broadcast("commit", tx_id, all_nodes)

            elif msg_type == "commit":
                self.committed.add(tx_id)

# ====================  PBFT  ====================
async def pbft_consensus(shard_nodes, all_nodes, tx_id):
    leader = shard_nodes[0]
    await leader.broadcast("preprepare", tx_id, all_nodes)

    while True:
        commit_count = sum(1 for n in shard_nodes if tx_id in n.committed)
        if commit_count >= 2:
            break
        await asyncio.sleep(0)

# ==================== ShardCutter  Relay ====================
async def cross_shard_consensus(src_shard, dst_shard, all_nodes, tx_id):
    await pbft_consensus(src_shard, all_nodes, tx_id)
    await pbft_consensus(dst_shard, all_nodes, tx_id)

# ====================  CTX  ====================
def generate_transactions(ctx, total):
    txs = []
    intra_num = int(total * (1 - ctx))
    cross_num = total - intra_num

    for _ in range(intra_num):
        s = random.randint(0, SHARD_COUNT - 1)
        txs.append(("intra", s, s))

    for _ in range(cross_num):
        s = random.randint(0, SHARD_COUNT - 1)
        d = random.randint(0, SHARD_COUNT - 1)
        while d == s:
            d = random.randint(0, SHARD_COUNT - 1)
        txs.append(("cross", s, d))

    random.shuffle(txs)
    return txs

def pack_blocks(txs):
    shard_tx = defaultdict(list)
    for tx in txs:
        _, _, dst_shard = tx
        shard_tx[dst_shard].append(tx)

    blocks = []
    height = {sid: 1 for sid in range(SHARD_COUNT)}

    for sid in range(SHARD_COUNT):
        batches = [
            shard_tx[sid][i:i + BLOCK_SIZE]
            for i in range(0, len(shard_tx[sid]), BLOCK_SIZE)
        ]
        for batch in batches:
            if batch:
                tx_type = batch[0][0]
                blocks.append(Block(sid, height[sid], batch, tx_type))
                height[sid] += 1
    return blocks

async def execute_block(block, shard_nodes, all_nodes):
    for tx in block.tx_list:
        tx_start = time.time()
        tx_type, src_sid, dst_sid = tx
        tx_id = random.getrandbits(64)

        if tx_type == "intra":
            await pbft_consensus(shard_nodes[src_sid], all_nodes, tx_id)
        else:
            await cross_shard_consensus(
                shard_nodes[src_sid],
                shard_nodes[dst_sid],
                all_nodes,
                tx_id
            )

        tx_end = time.time()
        latency_ms = (tx_end - tx_start) * 1000
        block.add_latency(latency_ms)

async def main():
    print("=" * 70)
    print("    ShardCutter =")
    print(f"    {SHARD_COUNT} | CTX{CTX:.2%} | {TOTAL_TX}")
    print("=" * 70)

    # 1. 
    txs = generate_transactions(CTX, TOTAL_TX)

    # 2. 
    blocks = pack_blocks(txs)
    print(f"[INFO] {len(blocks)}")

    # 3. 
    shard_nodes = {sid: [] for sid in range(SHARD_COUNT)}
    all_nodes = []
    for sid in range(SHARD_COUNT):
        for nid in range(NODES_PER_SHARD):
            node = Node(f"node_{sid}_{nid}", sid)
            shard_nodes[sid].append(node)
            all_nodes.append(node)

    # 4. 
    for node in all_nodes:
        node.task = asyncio.create_task(node.run(all_nodes))
    await asyncio.sleep(0.1)

    # 5. 
    print("[INFO]  & ...")
    start_total = time.time()

    tasks = [execute_block(b, shard_nodes, all_nodes) for b in blocks]
    await asyncio.gather(*tasks)

    end_total = time.time()
    elapsed = end_total - start_total

    # 6. 
    all_latencies = []
    for b in blocks:
        all_latencies.extend(b.tx_latencies)

    avg_latency = sum(all_latencies) / len(all_latencies)
    tps = TOTAL_TX / elapsed

    # 7. 
    print("\n" + "=" * 70)
    print("                    ")
    print(f"        : {TOTAL_TX}")
    print(f" CTX      : {CTX:.2%}")
    print(f"      : {elapsed:.3f} ")
    print(f"    : {avg_latency:.2f} ms")
    print(f" TPS        : {tps:.2f} tx/s")
    print("=" * 70)

    with open("shardcutter_blocks_result.json", "w", encoding="utf-8") as f:
        json.dump([b.to_dict() for b in blocks], f, indent=2)

    for node in all_nodes:
        await node.inbox.put(None)
    await asyncio.gather(*[node.task for node in all_nodes])

if __name__ == "__main__":
    asyncio.run(main())

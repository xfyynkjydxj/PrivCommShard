import asyncio
import random
import time
import json

CTX = 0.5017
TOTAL_TX = 10000
NODES_PER_SHARD = 4
SHARD_COUNT = 2
BLOCK_SIZE = 500
TIMEOUT = 1.0

class Block:
    def __init__(self, shard, height, txs, tx_type, latency):
        self.shard = shard
        self.height = height
        self.tx_count = len(txs)
        self.tx_type = tx_type
        self.latency_ms = round(latency * 1000, 2)
        self.timestamp = time.time()

    def to_dict(self):
        return {
            "shard": self.shard,
            "height": self.height,
            "tx_count": self.tx_count,
            "tx_type": self.tx_type,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp
        }

class Node:
    def __init__(self, node_id, shard_id):
        self.node_id = node_id
        self.shard_id = shard_id
        self.inbox = asyncio.Queue()
        self.committed = set()

    async def broadcast(self, msg_type, tx_id, all_nodes):
        for node in all_nodes:
            if node.shard_id == self.shard_id:
                await node.inbox.put((msg_type, tx_id))

    async def run(self, all_nodes):
        while True:
            msg = await self.inbox.get()
            if msg is None:
                break
            typ, tx_id = msg
            if typ == "preprepare":
                await self.broadcast("commit", tx_id, all_nodes)
            elif typ == "commit":
                self.committed.add(tx_id)

async def execute_batch(shard_nodes, all_nodes, tx_count):
    start = time.time()
    for _ in range(tx_count):
        tx_id = random.randint(1, 10**18)
        await shard_nodes[0].broadcast("preprepare", tx_id, all_nodes)
        while True:
            done = sum(1 for n in shard_nodes if tx_id in n.committed)
            if done >= 2:
                break
            await asyncio.sleep(0.00005)
    return time.time() - start

# ===================== 1.  CTX  =====================
def generate_transactions(ctx, total):
    intra_num = int(total * (1 - ctx))
    cross_num = total - intra_num
    txs = []

    for _ in range(intra_num):
        txs.append(("intra", random.randint(0, SHARD_COUNT-1)))

    for _ in range(cross_num):
        s1 = random.randint(0, SHARD_COUNT-1)
        s2 = random.randint(0, SHARD_COUNT-1)
        while s1 == s2:
            s2 = random.randint(0, SHARD_COUNT-1)
        txs.append(("cross", s1, s2))

    random.shuffle(txs)
    return txs

# ===================== 2.  =====================
def build_blocks(txs):
    intra_by_shard = {i: [] for i in range(SHARD_COUNT)}
    cross_by_shard = {i: [] for i in range(SHARD_COUNT)}

    for tx in txs:
        if tx[0] == "intra":
            sid = tx[1]
            intra_by_shard[sid].append(tx)
        else:
            sid = tx[1]
            cross_by_shard[sid].append(tx)

    blocks = []
    height = {i: 1 for i in range(SHARD_COUNT)}

    for sid, tx_list in intra_by_shard.items():
        for i in range(0, len(tx_list), BLOCK_SIZE):
            batch = tx_list[i:i+BLOCK_SIZE]
            blocks.append(("intra", sid, height[sid], batch))
            height[sid] += 1

    for sid, tx_list in cross_by_shard.items():
        for i in range(0, len(tx_list), BLOCK_SIZE):
            batch = tx_list[i:i+BLOCK_SIZE]
            blocks.append(("cross", sid, height[sid], batch))
            height[sid] += 1

    return blocks

# ===================== 3.  =====================
async def main():
    print("=" * 60)
    print("         - ")
    print("=" * 60)

    print(f"[1/5]  CTX={CTX:.2%}  {TOTAL_TX} ...")
    txs = generate_transactions(CTX, TOTAL_TX)

    print(f"[2/5]  {BLOCK_SIZE} ...")
    blocks = build_blocks(txs)
    print(f"  {len(blocks)} ")

    print(f"[3/5]  {SHARD_COUNT}   {NODES_PER_SHARD} ...")
    all_nodes = []
    shards = {i: [] for i in range(SHARD_COUNT)}
    for sid in range(SHARD_COUNT):
        for nid in range(NODES_PER_SHARD):
            node = Node(f"shard{sid}_node{nid}", sid)
            shards[sid].append(node)
            all_nodes.append(node)

    for node in all_nodes:
        asyncio.create_task(node.run(all_nodes))
    await asyncio.sleep(0.2)

    print(f"[4/5]  + ...")
    saved_blocks = []
    total_latency = 0

    for typ, sid, h, batch in blocks:
        latency = await execute_batch(shards[sid], all_nodes, len(batch))
        blk = Block(sid, h, batch, typ, latency)
        saved_blocks.append(blk)
        total_latency += latency

    print(f"[5/5]  blocks.json...")
    block_data = [b.to_dict() for b in saved_blocks]
    with open("blocks.json", "w", encoding="utf-8") as f:
        json.dump(block_data, f, indent=2)

    for node in all_nodes:
        await node.inbox.put(None)

    print("\n" + "=" * 60)
    print("                    ")
    print("=" * 60)
    print(f"{TOTAL_TX}")
    print(f" CTX{CTX:.2%}")
    print(f"{len(saved_blocks)}")
    print(f"{total_latency:.2f} s")
    print(f" TPS{TOTAL_TX / total_latency:.2f}")
    print(f"DYNASHARD_blocks.json")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())

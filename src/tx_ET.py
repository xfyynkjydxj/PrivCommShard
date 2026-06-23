import asyncio
import random
import time
import json
from collections import defaultdict

CTX = 0.5552
TOTAL_TX = 10000
SHARD_COUNT = 10           # 2/4/6/8/10 
NODES_PER_SHARD = 4
BLOCK_SIZE = 500
ACCOUNT_FILE = "estuary_accounts.json"

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

# ===================== PBFT =====================
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

async def execute_batch(shard_nodes, all_nodes, tx_num):
    start = time.time()
    for _ in range(tx_num):
        tx_id = random.randint(1, 10**18)
        await shard_nodes[0].broadcast("preprepare", tx_id, all_nodes)
        while True:
            done_num = sum(1 for n in shard_nodes if tx_id in n.committed)
            if done_num >= 2:
                break
            await asyncio.sleep(0.00005)
    return time.time() - start

def load_account_data():
    with open(ACCOUNT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_transactions(account_data, ctx, total):
    addr_list = list(account_data.keys())
    tx_list = []
    intra_num = int(total * (1 - ctx))
    cross_num = total - intra_num

    valid_shards = list(range(SHARD_COUNT))

    for _ in range(intra_num):
        addr = random.choice(addr_list)
        primary_s = account_data[addr]["primary_shard"]
        if primary_s not in valid_shards:
            primary_s = random.choice(valid_shards)
        tx_list.append(("intra", primary_s))

    for _ in range(cross_num):
        addr = random.choice(addr_list)
        primary_s = account_data[addr]["primary_shard"]
        if primary_s not in valid_shards:
            primary_s = random.choice(valid_shards)
        tx_list.append(("cross", primary_s))

    random.shuffle(tx_list)
    return tx_list

# =====================  KeyError =====================
def build_block_list(txs):
    shard_group = defaultdict(list)
    for tx in txs:
        _, sid = tx
        shard_group[sid].append(tx)

    blocks = []
    #   KeyError
    height_record = {i: 1 for i in range(SHARD_COUNT)}

    for sid in range(SHARD_COUNT):
        batch_list = shard_group.get(sid, [])
        for idx in range(0, len(batch_list), BLOCK_SIZE):
            batch = batch_list[idx:idx + BLOCK_SIZE]
            if len(batch) == 0:
                continue
            tx_type = batch[0][0]
            blocks.append((tx_type, sid, height_record[sid], batch))
            height_record[sid] += 1

    return blocks

async def main():
    print("=" * 60)
    print("    Estuary CSV +  + ")
    print("=" * 60)

    print("[1/5] ")
    acc_data = load_account_data()

    print(f"[2/5]  {TOTAL_TX} CTX={CTX:.2%}")
    txs = generate_transactions(acc_data, CTX, TOTAL_TX)

    print("[3/5] ")
    block_info = build_block_list(txs)
    print(f" : {len(block_info)} ")

    print(f"[4/5]  {SHARD_COUNT}  {NODES_PER_SHARD} ")
    all_nodes = []
    shard_nodes = {s: [] for s in range(SHARD_COUNT)}
    for sid in range(SHARD_COUNT):
        for nid in range(NODES_PER_SHARD):
            node = Node(f"est_{sid}_{nid}", sid)
            shard_nodes[sid].append(node)
            all_nodes.append(node)

    for node in all_nodes:
        asyncio.create_task(node.run(all_nodes))
    await asyncio.sleep(0.2)

    print("[5/5] ")
    saved_blocks = []
    total_latency = 0.0

    for typ, sid, h, batch in block_info:
        latency = await execute_batch(shard_nodes[sid], all_nodes, len(batch))
        blk = Block(sid, h, batch, typ, latency)
        saved_blocks.append(blk)
        total_latency += latency

    with open("estuary_blocks.json", "w", encoding="utf-8") as f:
        json.dump([b.to_dict() for b in saved_blocks], f, indent=2)

    for node in all_nodes:
        await node.inbox.put(None)

    print("\n" + "=" * 50)
    print("                ")
    print(f": {TOTAL_TX}")
    print(f" CTX: {CTX:.2%}")
    print(f": {total_latency:.2f} s")
    print(f" TPS: {TOTAL_TX / total_latency:.2f}")
    print(f": estuary_blocks.json")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())

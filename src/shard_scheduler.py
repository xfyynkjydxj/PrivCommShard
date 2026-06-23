import random
import hashlib


class ShardScheduler:

    def __init__(self, shard_num):
        self.K = shard_num

    # RandomShard
    def random_shard(self, accounts):

        mapping = {}

        for acc in accounts:
            mapping[acc] = random.randint(0, self.K - 1)

        return mapping

    # Monoxide
    def monoxide_shard(self, accounts):

        mapping = {}

        for acc in accounts:
            h = int(hashlib.md5(acc.encode()).hexdigest(), 16)
            mapping[acc] = h % self.K

        return mapping

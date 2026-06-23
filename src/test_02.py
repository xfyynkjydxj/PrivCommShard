import matplotlib.pyplot as plt
import numpy as np

shards = [4, 8, 16, 32, 64]

# SSI
ssi_shardcutter = [0.3097, 0.1938, 0.1575, 0.0894, 0.0628]
ssi_privcomm    = [0.9338, 0.9109, 0.9061, 0.9043, 0.9047]

colors = ['#1f77b4', '#2ca02c']
labels = ['ShardCutter', 'PrivCommShard']

plt.figure(figsize=(6, 4))
plt.plot(shards, ssi_shardcutter, marker='o', color=colors[0], label=labels[0])
plt.plot(shards, ssi_privcomm, marker='^', color=colors[1], label=labels[1])

plt.xlabel('Number of Shards')
plt.ylabel('SSI (Sharding Stability Index)')
plt.title('SSI vs. Number of Shards')
plt.xticks(shards)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('ssi_comparison.png', dpi=300)
plt.close()

print("SSI ssi_comparison.png")

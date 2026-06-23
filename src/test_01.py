import matplotlib.pyplot as plt
import numpy as np

shards = [4, 8, 16, 32, 64]

# CTX
ctx_shardcutter = [0.3373, 0.4658, 0.5095, 0.5526, 0.5794]
ctx_monoxide    = [0.7793, 0.8850, 0.9431, 0.9618, 0.9788]
ctx_privcomm    = [0.1036, 0.1154, 0.1277, 0.1283, 0.1291]

# la
lat_shardcutter = [1.78, 2.36, 3.17, 5.93, 17.86]
lat_monoxide    = [2.45, 2.90, 3.85, 7.50, 23.01]
lat_privcomm    = [1.49, 1.67, 2.22, 4.19, 12.52]

# TPS
tps_shardcutter = [11219.47, 8723.12, 6882.14, 5218.13, 3355.08]
tps_monoxide    = [8111.80, 7116.98, 5667.47, 4081.30, 2674.01]
tps_privcomm    = [13359.76, 12203.28, 9741.83, 7338.69, 4881.34]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
labels = ['ShardCutter', 'Monoxide', 'PrivCommShard']


plt.figure(figsize=(6, 4))
plt.plot(shards, ctx_shardcutter, marker='o', color=colors[0], label=labels[0])
plt.plot(shards, ctx_monoxide, marker='s', color=colors[1], label=labels[1])
plt.plot(shards, ctx_privcomm, marker='^', color=colors[2], label=labels[2])
plt.xlabel('Number of Shards')
plt.ylabel('CTX (Cross-Shard Transaction Rate)')
plt.title('CTX vs. Number of Shards')
plt.xticks(shards)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('ctx_comparison.png', dpi=300)
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(shards, lat_shardcutter, marker='o', color=colors[0], label=labels[0])
plt.plot(shards, lat_monoxide, marker='s', color=colors[1], label=labels[1])
plt.plot(shards, lat_privcomm, marker='^', color=colors[2], label=labels[2])
plt.xlabel('Number of Shards')
plt.ylabel('Latency (ms)')
plt.title('Latency vs. Number of Shards')
plt.xticks(shards)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('latency_comparison.png', dpi=300)
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(shards, tps_shardcutter, marker='o', color=colors[0], label=labels[0])
plt.plot(shards, tps_monoxide, marker='s', color=colors[1], label=labels[1])
plt.plot(shards, tps_privcomm, marker='^', color=colors[2], label=labels[2])
plt.xlabel('Number of Shards')
plt.ylabel('TPS (Transactions Per Second)')
plt.title('Throughput vs. Number of Shards')
plt.xticks(shards)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('tps_comparison.png', dpi=300)
plt.close()

print("ctx_comparison.png, latency_comparison.png, tps_comparison.png")

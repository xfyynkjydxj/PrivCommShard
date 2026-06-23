# PrivCommShard Experimental Code and Data

This repository is reserved for the experimental code and data release associated with the manuscript:

**PrivCommShard: a privacy-preserving dynamic sharding framework for consortium blockchain**

## Current Status

This repository does **not** contain article text, article source files, article figures, or bibliography files. It is intended only for experimental code, datasets, processed results, and reproducibility materials.

At the time of this cleanup, no executable experiment scripts, raw Ethereum trace files, processed datasets, or result tables were available for upload.

## Planned Repository Structure

When the experimental materials are available, use the following structure:

```text
PrivCommShard/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── raw/                 # raw or externally linked Ethereum traces
│   └── processed/           # processed transaction-window tables
├── src/
│   ├── graph_construction.py
│   ├── edge_perturbation.py
│   ├── temporal_fusion.py
│   ├── community_detection.py
│   ├── shard_assignment.py
│   └── metrics.py
├── experiments/
│   ├── run_baselines.py
│   ├── run_privcommshard.py
│   └── parameter_sweep.py
└── results/
    ├── tables/
    └── figures/
```

## Dataset Information

The manuscript reports experiments on Ethereum transaction traces containing more than 100,000 transaction records. The evaluated shard configurations are `K = {4, 8, 16, 32, 64}`, with a default transaction-window size of 10,000 transactions and 10 consecutive windows.

The dataset files are not currently included in this repository. Before using this repository as a formal reproducibility archive, add one of the following:

1. the raw and processed dataset files, if redistribution is permitted; or
2. a persistent public dataset link and scripts for preprocessing the downloaded data.

## Code Information

The code release should include implementations for:

1. weighted account-interaction graph construction;
2. edge perturbation controlled by `epsilon`;
3. temporal graph fusion controlled by `lambda`;
4. Louvain-based community detection;
5. history-aware shard assignment;
6. CTX, NMI, ESR, SSI, latency, and TPS calculation;
7. baseline comparison with Monoxide and ShardCutter;
8. parameter-sweep experiments.

No executable code is currently included because the local submission materials did not contain the experiment implementation.

## Usage Instructions

After code and data are added, the repository should provide commands such as:

```bash
python -m venv .venv
pip install -r requirements.txt
python experiments/run_privcommshard.py --config configs/default.yaml
python experiments/run_baselines.py --config configs/default.yaml
python experiments/parameter_sweep.py --config configs/parameter_sweep.yaml
```

These commands are placeholders until the actual implementation is added.

## Requirements

Requirements should be specified in `requirements.txt` when the code is released. Expected dependencies may include:

- Python 3.10 or later
- NumPy
- pandas
- NetworkX
- python-louvain or another Louvain implementation
- matplotlib

## Citation

If using the future code or dataset release, cite the associated manuscript after publication.

## License

No license is currently assigned. Add a `LICENSE` file before releasing executable code or redistributable data.

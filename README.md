# PrivCommShard Experimental Code and Data

This repository contains the experimental code and release data for PrivCommShard, a dynamic sharding framework for consortium blockchain experiments.

No manuscript source files are included in this repository. The repository is intended for code and data review only.

## Repository Contents

- `src/`: Python scripts for transaction graph construction, sharding baselines, PrivCommShard experiments, parameter analysis, and consensus/throughput simulation.
- `data/ethereum_clean.txt`: cleaned Ethereum transaction dataset used by the scripts. The file is tab-separated and contains the columns `Unnamed: 0`, `TxHash`, `BlockHeight`, `TimeStamp`, `From`, `To`, `Value`, and `isError`.
- `data/experiment_results*.csv`: parameter-sweep outputs for different shard counts.
- `data/nmi_results*.csv`: NMI evaluation outputs.
- `data/experiment_smoothed.csv`: smoothed result table used for plotting.
- `results/figures/`: generated result figures.

Large intermediate files and raw dumps are not included because they exceed normal GitHub repository limits and are not required to run the released scripts.

## Requirements

Python 3.9 or later is recommended.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Run commands from the repository root. The scripts use relative paths such as `../data/...`, so execute experiment scripts from the `src` directory:

```bash
cd src
python run_p_c_3d.py
python tx_SC.py
python test_01.py
python test_02.py
```

Additional baseline and analysis scripts are also available in `src/`, including:

- `run_CommunityShard.py`
- `run_DYNASHARD.py`
- `run_Estuary.py`
- `run_Monoxide.py`
- `run_ShardCutter.py`
- `run_Temporal.py`
- `run_nmi.py`
- `run_data.py`
- `run_3d.py`

## Dataset Notes

The released transaction file is `data/ethereum_clean.txt`. It is tab-separated and has already been filtered for successful transactions, non-empty addresses, non-self-loop transfers, and chronological order.

If rebuilding the cleaned file from a raw dump, place the raw input as `data/ethereum_raw.csv` and run:

```bash
cd src
python preprocess.py
```

The raw dump is not included in this repository.

## Method Overview

The main experiment scripts build transaction graphs from windowed blockchain transactions, apply structural perturbation and temporal graph fusion, detect communities, assign communities to shards, and report metrics such as CTX, SSI, ERR/RA, NMI, latency, and TPS.

Baseline scripts provide comparable runs for static hashing, community-based sharding, temporal sharding, ShardCutter-style partitioning, Estuary-style assignment, and DynaShard-style assignment.

## Citation

If this code or dataset is used in a publication, cite the associated PrivCommShard manuscript or repository record.

## License

No explicit open-source license is provided yet. Contact the repository owner before reusing the code or dataset outside review and reproducibility purposes.

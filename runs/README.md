# 🚀 OutMind Runs: End-to-End Pipeline

The `runs/` directory contains automated, single-command pipelines that run the full OutMind model lifecycle from scratch:

```
Download Dataset (HF) ➔ Train Tokenizer ➔ Eval Tokenizer ➔ Pretrain ➔ Eval Pretrain ➔ SFT ➔ Eval SFT
```

---

## 📊 Model Tiers & Hardware Specs

| Script | Model Tier | Total Params | Active Layers | Dim / FFN | Recommended Hardware | Minimum VRAM / RAM |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| [`run_nano.py`](file:///runs/run_nano.py) | **Nano** | **~34M** | 6 | 384 / 1024 | Laptop CPU, Apple Silicon, GTX 1650 | 2 GB |
| [`run_small.py`](file:///runs/run_small.py) | **Small** | **~125M** | 12 | 768 / 2048 | Single GPU (RTX 3060/4060, Colab T4) | 6 GB |
| [`run_medium.py`](file:///runs/run_medium.py) | **Medium** | **~246M** | 16 | 1024 / 2816 | Workstation (RTX 3080/4080, A10G) | 12 GB |
| [`run_large.py`](file:///runs/run_large.py) | **Large** | **~693M** | 24 | 1536 / 4096 | Flagship (RTX 3090/4090, A100, H100) | 24 GB |

---

## ⚡ Quick Start (1-Command Execution)

### 1. Test Verification (`--mode quick`)
Runs a fast cycle (~1-2 minutes) on a small dataset shard to verify everything works properly:

```bash
# Python
python runs/run_nano.py --mode quick

# Windows PowerShell
.\runs\run.ps1 nano --mode quick

# Linux / Colab / macOS Bash
bash runs/run.sh nano --mode quick
```

### 2. Full Production Training (`--mode full`)
Runs the complete dataset training across pretrain and instruction tuning:

```bash
# Nano (34M)
python runs/run_nano.py --mode full

# Small (125M)
python runs/run_small.py --mode full

# Medium (246M)
python runs/run_medium.py --mode full

# Large (693M)
python runs/run_large.py --mode full
```

---

## 🛠️ CLI Options & Customization

All scripts support the following parameters:

```bash
python runs/run_nano.py \
    --mode full \
    --device cuda \
    --batch-size 8 \
    --pretrain-steps 2000 \
    --sft-steps 500 \
    --checkpoint-dir checkpoints
```

* `--mode [quick|full]`: `quick` runs rapid verification with few steps; `full` runs complete training.
* `--device [auto|cuda|cpu]`: Auto-detects NVIDIA CUDA GPU, Apple Silicon MPS, or CPU fallback.
* `--pretrain-steps [N]`: Custom number of pretraining steps.
* `--sft-steps [N]`: Custom number of SFT steps.
* `--batch-size [N]`: Batch size per step.
* `--force-download`: Re-download dataset shards from Hugging Face (`Mesosfer/outmind-dataset`).
* `--retrain-tokenizer`: Force retraining of BPE tokenizer even if one already exists in `data/processed/`.

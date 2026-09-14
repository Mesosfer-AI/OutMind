"""Master End-to-End Execution Pipeline for OutMind.

Executes the complete 7-stage SLM lifecycle in a single command:
  1. Download / Sync dataset from Hugging Face (Mesosfer/outmind-dataset)
  2. Train custom BPE Tokenizer
  3. Evaluate Tokenizer (fertility, compression ratio, round-trip fidelity)
  4. Pretrain OutMind (Autoregressive causal language modeling)
  5. Evaluate Pretrained Model (validation perplexity and prompt completion)
  6. Supervised Fine-Tuning (SFT with dynamic prompt masking on ChatML)
  7. Evaluate SFT Model (multi-turn instruction following and benchmark check)
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is in sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader

from outmind.config import OutMindConfig
from outmind.dataset import PretrainDataset, SFTDataset
from outmind.generate import generate
from outmind.model import OutMindForCausalLM
from outmind.tokenizer import OutMindTokenizer
from outmind.trainer import Trainer, TrainerConfig
from scripts.eval import eval_pretrain_perplexity, eval_tokenizer
from scripts.train_pretrain import load_tokens_from_path
from scripts.train_sft import load_dialogues_from_path
from scripts.train_tokenizer import train_bpe

DEFAULT_DATASET_REPO = "Mesosfer/outmind-dataset"


def log_phase(step: int, total: int, title: str):
    banner = f"== [Phase {step}/{total}] {title} =="
    print("\n" + "=" * len(banner))
    print(banner)
    print("=" * len(banner) + "\n", flush=True)


# -----------------------------------------------------------------------------
# 1. Download Dataset
# -----------------------------------------------------------------------------
def step_download_data(
    repo_id: str = DEFAULT_DATASET_REPO,
    local_dir: str = "data/cleaned",
    force_download: bool = False,
    quick_mode: bool = False,
) -> str:
    log_phase(1, 7, f"Checking / Downloading Dataset ({repo_id})")

    pretrain_dir = os.path.join(local_dir, "pretrain", "train")
    sft_dir = os.path.join(local_dir, "sft", "train")

    has_local_data = (
        os.path.exists(pretrain_dir)
        and any(f.endswith(".parquet") for f in os.listdir(pretrain_dir))
        and os.path.exists(sft_dir)
        and any(f.endswith(".parquet") for f in os.listdir(sft_dir))
    )

    if has_local_data and not force_download:
        print(f"[OutMind Pipeline] Local dataset verified in '{local_dir}'. Skipping download.")
        return local_dir

    print(f"[OutMind Pipeline] Downloading dataset from Hugging Face '{repo_id}' to '{local_dir}'...")
    from huggingface_hub import snapshot_download

    allow_patterns = None
    if quick_mode:
        print("[OutMind Pipeline] Quick mode enabled: downloading minimal subset of parquet shards...")
        allow_patterns = [
            "pretrain/train/*0000*.parquet",
            "pretrain/train/*0001*.parquet",
            "pretrain/val/*.parquet",
            "sft/train/*0000*.parquet",
            "sft/train/*0001*.parquet",
            "sft/val/*.parquet",
            "validation/*.jsonl",
            "tokenizer/*",
        ]

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=allow_patterns,
    )
    print(f"[OutMind Pipeline] Dataset successfully downloaded to '{local_dir}'.")
    return local_dir


# -----------------------------------------------------------------------------
# 2. Train Tokenizer
# -----------------------------------------------------------------------------
def step_train_tokenizer(
    data_dir: str,
    output_prefix: str = "data/processed/outmind_tokenizer",
    vocab_size: int = 64000,
    max_samples: int = 50000,
    skip_if_exists: bool = True,
) -> str:
    log_phase(2, 7, f"Training BPE Tokenizer (vocab: {vocab_size:,})")

    model_path = f"{output_prefix}.model"
    if skip_if_exists and os.path.exists(model_path):
        print(f"[OutMind Pipeline] Tokenizer model already exists at '{model_path}'. Skipping training.")
        return model_path

    pretrain_train_dir = os.path.join(data_dir, "pretrain", "train")
    if not os.path.exists(pretrain_train_dir):
        raise FileNotFoundError(f"Pretrain data not found at {pretrain_train_dir}")

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    t0 = time.time()
    train_bpe(
        parquet_dir=pretrain_train_dir,
        output_prefix=output_prefix,
        vocab_size=vocab_size,
        max_samples=max_samples,
    )
    print(f"[OutMind Pipeline] Tokenizer trained in {time.time() - t0:.2f}s -> '{model_path}'.")
    return model_path


# -----------------------------------------------------------------------------
# 3. Evaluate Tokenizer
# -----------------------------------------------------------------------------
def step_eval_tokenizer(tokenizer_path: str):
    log_phase(3, 7, "Evaluating Tokenizer Quality & SOTA Comparison")
    tok = OutMindTokenizer(vocab_file=tokenizer_path)
    print(f"[OutMind Pipeline] Loaded Tokenizer: Vocab Size = {tok.vocab_size:,}")
    eval_tokenizer(tok, compare_baselines=True)


# -----------------------------------------------------------------------------
# 4. Pretrain Model
# -----------------------------------------------------------------------------
def step_pretrain(
    preset: str,
    data_dir: str,
    tokenizer_path: str,
    max_steps: int = 500,
    batch_size: int = 2,
    device: str = "auto",
    checkpoint_dir: str = "checkpoints",
    max_tokens: Optional[int] = None,
) -> str:
    log_phase(4, 7, f"Pretraining OutMind-{preset.upper()} ({max_steps} steps)")

    tok = OutMindTokenizer(vocab_file=tokenizer_path)
    pretrain_data_path = os.path.join(data_dir, "pretrain", "train")

    print(f"[OutMind Pipeline] Tokenizing pretrain stream from '{pretrain_data_path}'...")
    tokens = load_tokens_from_path(pretrain_data_path, tok, max_tokens=max_tokens)
    print(f"[OutMind Pipeline] Loaded {len(tokens):,} pretraining tokens.")

    cfg = getattr(OutMindConfig, preset)(vocab_size=tok.vocab_size)
    dataset = PretrainDataset(tokens, max_seq_len=cfg.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    out_ckpt = os.path.join(checkpoint_dir, f"{preset}_pretrain.pt")
    trainer_cfg = TrainerConfig(
        max_steps=max_steps,
        save_every=max(1, max_steps // 2),
        checkpoint_dir=checkpoint_dir,
        device=device,
    )

    model = OutMindForCausalLM(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OutMind Pipeline] Architecture: {preset} | Total Parameters: {total_params:,}")

    trainer = Trainer(model, trainer_cfg)
    trainer.train(dataloader)
    trainer.save_checkpoint(max_steps, path=out_ckpt)
    print(f"[OutMind Pipeline] Pretraining completed. Saved checkpoint to '{out_ckpt}'.")
    return out_ckpt


# -----------------------------------------------------------------------------
# 5. Evaluate Pretrained Model
# -----------------------------------------------------------------------------
def step_eval_pretrain(
    checkpoint_path: str,
    preset: str,
    data_dir: str,
    tokenizer_path: str,
    device: str = "auto",
):
    log_phase(5, 7, "Evaluating Pretrained Model Perplexity & Generation")

    tok = OutMindTokenizer(vocab_file=tokenizer_path)
    cfg = getattr(OutMindConfig, preset)(vocab_size=tok.vocab_size)
    model = OutMindForCausalLM(cfg)

    dev = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
    ckpt = torch.load(checkpoint_path, map_location=dev)
    state = ckpt.get("model_state") or ckpt.get("model_state_dict") or ckpt
    model.load_state_dict(state)
    model.to(dev)
    model.eval()

    val_dir = os.path.join(data_dir, "pretrain", "val")
    if os.path.exists(val_dir):
        import pyarrow.parquet as pq
        val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith(".parquet")]
        eval_lines = []
        for vf in val_files[:2]:
            pfile = pq.ParquetFile(vf)
            for batch in pfile.iter_batches(batch_size=200, columns=["text"]):
                for t in batch["text"].to_pylist():
                    if t and len(t) > 50:
                        eval_lines.append(t)
                        if len(eval_lines) >= 50:
                            break
                if len(eval_lines) >= 50:
                    break
        if eval_lines:
            ppl = eval_pretrain_perplexity(model, tok, eval_lines, device=dev)
            print(f"[OutMind Pipeline] Validation Perplexity (PPL): {ppl:.2f}")

    # Qualitative completion test
    test_prompt = "Kecerdasan buatan (AI) adalah teknologi yang"
    output = generate(model, tok, test_prompt, max_new_tokens=30, temperature=0.7, device=dev)
    print("\n[OutMind Pipeline] Sample Completion Test:")
    print(f"Prompt: {test_prompt}")
    print(f"Generation: {output}\n")


# -----------------------------------------------------------------------------
# 6. SFT Instruction Tuning
# -----------------------------------------------------------------------------
def step_train_sft(
    checkpoint_path: str,
    preset: str,
    data_dir: str,
    tokenizer_path: str,
    max_steps: int = 250,
    batch_size: int = 2,
    device: str = "auto",
    checkpoint_dir: str = "checkpoints",
    max_samples: Optional[int] = None,
) -> str:
    log_phase(6, 7, f"Supervised Fine-Tuning (SFT) on ChatML ({max_steps} steps)")

    tok = OutMindTokenizer(vocab_file=tokenizer_path)
    sft_data_path = os.path.join(data_dir, "sft", "train")

    print(f"[OutMind Pipeline] Loading ChatML dialogues from '{sft_data_path}'...")
    dialogues = load_dialogues_from_path(sft_data_path, max_samples=max_samples)
    print(f"[OutMind Pipeline] Loaded {len(dialogues):,} conversation dialogues.")

    cfg = getattr(OutMindConfig, preset)(vocab_size=tok.vocab_size)
    dataset = SFTDataset(dialogues, tok, max_seq_len=cfg.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    out_ckpt = os.path.join(checkpoint_dir, f"{preset}_sft.pt")
    trainer_cfg = TrainerConfig(
        max_steps=max_steps,
        save_every=max(1, max_steps // 2),
        checkpoint_dir=checkpoint_dir,
        device=device,
    )

    model = OutMindForCausalLM(cfg)
    dev = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state") or ckpt.get("model_state_dict") or ckpt
        model.load_state_dict(state, strict=False)
        print(f"[OutMind Pipeline] Successfully transferred weights from pretrain base: '{checkpoint_path}'")

    trainer = Trainer(model, trainer_cfg)
    trainer.train(dataloader)
    trainer.save_checkpoint(max_steps, path=out_ckpt)
    print(f"[OutMind Pipeline] SFT completed. Saved checkpoint to '{out_ckpt}'.")
    return out_ckpt


# -----------------------------------------------------------------------------
# 7. Evaluate SFT Model
# -----------------------------------------------------------------------------
def step_eval_sft(
    checkpoint_path: str,
    preset: str,
    tokenizer_path: str,
    device: str = "auto",
):
    log_phase(7, 7, "Evaluating Fine-Tuned (SFT) Model & Multi-turn Chat")

    tok = OutMindTokenizer(vocab_file=tokenizer_path)
    cfg = getattr(OutMindConfig, preset)(vocab_size=tok.vocab_size)
    model = OutMindForCausalLM(cfg)

    dev = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
    ckpt = torch.load(checkpoint_path, map_location=dev)
    state = ckpt.get("model_state") or ckpt.get("model_state_dict") or ckpt
    model.load_state_dict(state)
    model.to(dev)
    model.eval()

    test_queries = [
        "Jelaskan apa itu OutMind dalam satu kalimat singkat!",
        "Tulis fungsi Python untuk menghitung bilangan prima.",
        "Hitunglah 15 * 8 + 40.",
    ]

    print("[OutMind Pipeline] Multi-Turn Chat Evaluation Samples:\n")
    for q in test_queries:
        formatted_prompt = (
            f"<|start_header_id|>system<|end_header_id|>\n"
            f"You are OutMind, a helpful Indonesian-centric AI assistant.<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"{q}<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )
        resp = generate(model, tok, formatted_prompt, max_new_tokens=48, temperature=0.7, device=dev)
        print(f"User: {q}")
        # Strip system prompt header for clean readability
        clean_resp = resp.split("<|start_header_id|>assistant<|end_header_id|>\n")[-1]
        print(f"OutMind: {clean_resp.strip()}\n" + "-" * 40)

    print("\n[OutMind Pipeline] End-to-end run completed with full lifecycle verification!")


# -----------------------------------------------------------------------------
# Main Orchestrator
# -----------------------------------------------------------------------------
def run_pipeline(
    preset: str = "nano",
    mode: str = "quick",
    pretrain_steps: Optional[int] = None,
    sft_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: str = "auto",
    force_download: bool = False,
    skip_tokenizer_train: bool = True,
    checkpoint_dir: str = "checkpoints",
):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n" + "=" * 60)
    print(f"[OutMind] End-to-End Run: Preset = {preset.upper()} | Mode = {mode.upper()}")
    print("=" * 60 + "\n")

    is_quick = mode == "quick"

    # Step defaults based on mode
    if pretrain_steps is None:
        pretrain_steps = 10 if is_quick else (500 if preset == "nano" else 2000)
    if sft_steps is None:
        sft_steps = 5 if is_quick else (250 if preset == "nano" else 1000)
    if batch_size is None:
        batch_size = 2 if is_quick else (4 if preset in ["nano", "small"] else 8)

    max_tokens = 50000 if is_quick else None
    max_sft_samples = 200 if is_quick else None

    # 1. Download / Verify Data
    data_dir = step_download_data(
        force_download=force_download,
        quick_mode=is_quick,
    )

    # 2. Train Tokenizer
    tok_path = step_train_tokenizer(
        data_dir=data_dir,
        output_prefix="data/processed/outmind_tokenizer",
        vocab_size=64000,
        max_samples=25000 if is_quick else 100000,
        skip_if_exists=skip_tokenizer_train,
    )

    # 3. Evaluate Tokenizer
    step_eval_tokenizer(tok_path)

    # 4. Pretrain Model
    pretrain_ckpt = step_pretrain(
        preset=preset,
        data_dir=data_dir,
        tokenizer_path=tok_path,
        max_steps=pretrain_steps,
        batch_size=batch_size,
        device=device,
        checkpoint_dir=checkpoint_dir,
        max_tokens=max_tokens,
    )

    # 5. Evaluate Pretrained Model
    step_eval_pretrain(
        checkpoint_path=pretrain_ckpt,
        preset=preset,
        data_dir=data_dir,
        tokenizer_path=tok_path,
        device=device,
    )

    # 6. Supervised Fine-Tuning (SFT)
    sft_ckpt = step_train_sft(
        checkpoint_path=pretrain_ckpt,
        preset=preset,
        data_dir=data_dir,
        tokenizer_path=tok_path,
        max_steps=sft_steps,
        batch_size=batch_size,
        device=device,
        checkpoint_dir=checkpoint_dir,
        max_samples=max_sft_samples,
    )

    # 7. Evaluate SFT Model
    step_eval_sft(
        checkpoint_path=sft_ckpt,
        preset=preset,
        tokenizer_path=tok_path,
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(description="OutMind Unified End-to-End Run Pipeline")
    parser.add_argument("--preset", type=str, choices=["nano", "small", "medium", "large", "moe"], default="nano")
    parser.add_argument("--mode", type=str, choices=["quick", "full"], default="quick", help="quick (fast test) or full")
    parser.add_argument("--pretrain-steps", type=int, default=None)
    parser.add_argument("--sft-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--force-download", action="store_true", help="Force redownload from Hugging Face")
    parser.add_argument("--retrain-tokenizer", action="store_true", help="Force retrain tokenizer even if exists")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    run_pipeline(
        preset=args.preset,
        mode=args.mode,
        pretrain_steps=args.pretrain_steps,
        sft_steps=args.sft_steps,
        batch_size=args.batch_size,
        device=args.device,
        force_download=args.force_download,
        skip_tokenizer_train=not args.retrain_tokenizer,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()

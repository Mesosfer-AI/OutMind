"""CLI script to run Stage 1 Pretraining on raw text."""

import argparse
import os
import torch
from torch.utils.data import DataLoader

from typing import List, Optional

from outmind.config import OutMindConfig
from outmind.dataset import PretrainDataset
from outmind.model import OutMindForCausalLM
from outmind.tokenizer import OutMindTokenizer
from outmind.trainer import Trainer, TrainerConfig


def load_tokens_from_path(data_path: str, tokenizer: OutMindTokenizer, max_tokens: Optional[int] = None) -> List[int]:
    """Loads token stream from raw text file, parquet file, or parquet directory with memory budgeting."""
    if os.path.isdir(data_path) or data_path.endswith(".parquet"):
        import pyarrow.parquet as pq
        parquet_files = []
        if os.path.isdir(data_path):
            for root, _, files in os.walk(data_path):
                for f in files:
                    if f.endswith(".parquet"):
                        parquet_files.append(os.path.join(root, f))
            parquet_files.sort()
        else:
            parquet_files = [data_path]

        tokens: List[int] = []
        last_logged = 0
        for pf in parquet_files:
            pfile = pq.ParquetFile(pf)
            for batch in pfile.iter_batches(batch_size=1000, columns=["text"]):
                for text in batch["text"].to_pylist():
                    if text:
                        tokens.extend(tokenizer.encode(text))
                        if len(tokens) - last_logged >= 5_000_000:
                            last_logged = len(tokens)
                            target_str = f" / {max_tokens:,}" if max_tokens else ""
                            print(f"[OutMind] Streamed {len(tokens):,}{target_str} tokens into memory...", flush=True)
                        if max_tokens and len(tokens) >= max_tokens:
                            return tokens[:max_tokens]
        return tokens
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = tokenizer.encode(text)
        return tokens[:max_tokens] if max_tokens else tokens


def main():
    parser = argparse.ArgumentParser(description="OutMind Stage 1: Pretraining")
    parser.add_argument("--data-path", type=str, default=None, help="Path to raw text corpus or parquet dir")
    parser.add_argument("--preset", type=str, default="nano", choices=["nano", "small", "medium", "large", "moe"])
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dummy-test", action="store_true", help="Run 5 steps on dummy data for verification")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--output-model", type=str, default=None, help="Final checkpoint save path")
    args = parser.parse_args()

    tokenizer = OutMindTokenizer()

    if args.dummy_test:
        text = "Halo! Ini adalah pengujian pretraining OutMind untuk bahasa Indonesia dan kode. " * 50
        tokens = tokenizer.encode(text)
        cfg = OutMindConfig.nano(vocab_size=max(tokens) + 1, max_seq_len=64, n_layers=2, d_model=128, n_heads=4, n_kv_heads=2)
        dataset = PretrainDataset(tokens, max_seq_len=64)
        trainer_cfg = TrainerConfig(max_steps=5, warmup_steps=1, save_every=5, checkpoint_dir=args.checkpoint_dir)
    else:
        if not args.data_path or not os.path.exists(args.data_path):
            raise FileNotFoundError(f"Data path not found: {args.data_path}")
        cfg = getattr(OutMindConfig, args.preset)(vocab_size=tokenizer.vocab_size)
        needed_tokens = int(args.max_steps * args.batch_size * (cfg.max_seq_len + 1) * 1.25)
        tokens = load_tokens_from_path(args.data_path, tokenizer, max_tokens=needed_tokens)
        dataset = PretrainDataset(tokens, max_seq_len=cfg.max_seq_len)
        trainer_cfg = TrainerConfig(max_steps=args.max_steps, checkpoint_dir=args.checkpoint_dir)

    model = OutMindForCausalLM(cfg)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    trainer = Trainer(model, trainer_cfg)

    print(f"[OutMind] Starting pretraining ({args.preset} preset, {len(dataset)} chunks)...")
    trainer.train(dataloader)
    print("[OutMind] Pretraining completed successfully.")
    if args.output_model:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_model)), exist_ok=True)
        trainer.save_checkpoint(args.max_steps, path=args.output_model)
        print(f"[OutMind] Saved final pretrain checkpoint to {args.output_model}")


if __name__ == "__main__":
    main()

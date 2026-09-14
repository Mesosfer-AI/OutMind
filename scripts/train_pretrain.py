"""CLI script to run Stage 1 Pretraining on raw text."""

import argparse
import os
import torch
from torch.utils.data import DataLoader

from outmind.config import OutMindConfig
from outmind.dataset import PretrainDataset
from outmind.model import OutMindForCausalLM
from outmind.tokenizer import OutMindTokenizer
from outmind.trainer import Trainer, TrainerConfig


def main():
    parser = argparse.ArgumentParser(description="OutMind Stage 1: Pretraining")
    parser.add_argument("--data-path", type=str, default=None, help="Path to raw text corpus")
    parser.add_argument("--preset", type=str, default="nano", choices=["nano", "small", "moe"])
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dummy-test", action="store_true", help="Run 5 steps on dummy data for verification")
    args = parser.parse_args()

    tokenizer = OutMindTokenizer()

    if args.dummy_test:
        text = "Halo! Ini adalah pengujian pretraining OutMind untuk bahasa Indonesia dan kode. " * 50
        tokens = tokenizer.encode(text)
        cfg = OutMindConfig.nano(vocab_size=max(tokens) + 1, max_seq_len=64, n_layers=2, d_model=128, n_heads=4, n_kv_heads=2)
        dataset = PretrainDataset(tokens, max_seq_len=64)
        trainer_cfg = TrainerConfig(max_steps=5, warmup_steps=1, save_every=5)
    else:
        if not args.data_path or not os.path.exists(args.data_path):
            raise FileNotFoundError(f"Data path not found: {args.data_path}")
        with open(args.data_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = tokenizer.encode(text)
        cfg = getattr(OutMindConfig, args.preset)(vocab_size=tokenizer.vocab_size)
        dataset = PretrainDataset(tokens, max_seq_len=cfg.max_seq_len)
        trainer_cfg = TrainerConfig(max_steps=args.max_steps)

    model = OutMindForCausalLM(cfg)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    trainer = Trainer(model, trainer_cfg)

    print(f"[OutMind] Starting pretraining ({args.preset} preset, {len(dataset)} chunks)...")
    trainer.train(dataloader)
    print("[OutMind] Pretraining completed successfully.")


if __name__ == "__main__":
    main()

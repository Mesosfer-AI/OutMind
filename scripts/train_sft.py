"""CLI script to run Stage 2 Supervised Fine-Tuning (SFT) on chat datasets."""

import argparse
import json
import os
import torch
from torch.utils.data import DataLoader

from outmind.config import OutMindConfig
from outmind.dataset import SFTDataset
from outmind.lora import apply_lora
from outmind.model import OutMindForCausalLM
from outmind.tokenizer import OutMindTokenizer
from outmind.trainer import Trainer, TrainerConfig


def main():
    parser = argparse.ArgumentParser(description="OutMind Stage 2: Instruction SFT")
    parser.add_argument("--data-path", type=str, default=None, help="Path to JSONL conversation data")
    parser.add_argument("--use-lora", action="store_true", help="Enable parameter-efficient fine-tuning with LoRA")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dummy-test", action="store_true", help="Run 5 steps on dummy conversation data")
    args = parser.parse_args()

    tokenizer = OutMindTokenizer()

    if args.dummy_test:
        dummy_dialogues = [
            [
                {"role": "system", "content": "Kamu adalah asisten OutMind yang ramah."},
                {"role": "user", "content": "Siapa namamu?"},
                {"role": "assistant", "content": "Saya adalah OutMind, AI cerdas buatan Mesosfer-AI."},
            ],
            [
                {"role": "system", "content": "Kamu adalah asisten OutMind yang ramah."},
                {"role": "user", "content": "Apa ibu kota Indonesia?"},
                {"role": "assistant", "content": "Ibu kota Indonesia adalah Nusantara (IKN)."},
            ],
        ] * 10
        dataset = SFTDataset(dummy_dialogues, tokenizer, max_seq_len=64)
        max_id = max(sample["input_ids"].max().item() for sample in dataset)
        cfg = OutMindConfig.nano(vocab_size=max_id + 1, max_seq_len=64, n_layers=2, d_model=128, n_heads=4, n_kv_heads=2)
        trainer_cfg = TrainerConfig(max_steps=5, warmup_steps=1, save_every=5)
    else:
        if not args.data_path or not os.path.exists(args.data_path):
            raise FileNotFoundError(f"SFT data path not found: {args.data_path}")
        dialogues = []
        with open(args.data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                dialogues.append(item["messages"])
        cfg = OutMindConfig.nano(vocab_size=tokenizer.vocab_size)
        dataset = SFTDataset(dialogues, tokenizer, max_seq_len=cfg.max_seq_len)
        trainer_cfg = TrainerConfig(max_steps=args.max_steps)

    model = OutMindForCausalLM(cfg)

    if args.use_lora:
        adapters = apply_lora(model, rank=8, alpha=16.0)
        print(f"[OutMind] LoRA activated: injected {len(adapters)} low-rank adapters.")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    trainer = Trainer(model, trainer_cfg)

    print(f"[OutMind] Starting SFT ({len(dataset)} conversation samples)...")
    trainer.train(dataloader)
    print("[OutMind] SFT completed successfully.")


if __name__ == "__main__":
    main()

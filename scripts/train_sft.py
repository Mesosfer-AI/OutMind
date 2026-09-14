"""CLI script to run Stage 2 Supervised Fine-Tuning (SFT) on chat datasets."""

import argparse
import json
import os
import torch
from torch.utils.data import DataLoader

from typing import Dict, List, Optional

from outmind.config import OutMindConfig
from outmind.dataset import SFTDataset
from outmind.lora import apply_lora
from outmind.model import OutMindForCausalLM
from outmind.tokenizer import OutMindTokenizer
from outmind.trainer import Trainer, TrainerConfig


def load_dialogues_from_path(data_path: str, max_samples: Optional[int] = None) -> List[List[Dict[str, str]]]:
    """Loads ChatML dialogues from JSONL file, parquet file, or parquet directory."""
    dialogues: List[List[Dict[str, str]]] = []
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

        for pf in parquet_files:
            pfile = pq.ParquetFile(pf)
            for batch in pfile.iter_batches(batch_size=1000, columns=["messages"]):
                for raw_msg in batch["messages"].to_pylist():
                    if not raw_msg:
                        continue
                    msgs = json.loads(raw_msg) if isinstance(raw_msg, str) else raw_msg
                    dialogues.append(msgs)
                    if max_samples and len(dialogues) >= max_samples:
                        return dialogues
        return dialogues
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    dialogues.append(item["messages"])
                    if max_samples and len(dialogues) >= max_samples:
                        break
        return dialogues


def main():
    parser = argparse.ArgumentParser(description="OutMind Stage 2: Instruction SFT")
    parser.add_argument("--data-path", type=str, default=None, help="Path to JSONL conversation data or parquet dir")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to pretrained model checkpoint (.pt)")
    parser.add_argument("--preset", type=str, default="nano", choices=["nano", "small", "medium", "large", "moe"])
    parser.add_argument("--use-lora", action="store_true", help="Enable parameter-efficient fine-tuning with LoRA")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dummy-test", action="store_true", help="Run 5 steps on dummy conversation data")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--output-model", type=str, default=None, help="Final checkpoint save path")
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
        trainer_cfg = TrainerConfig(max_steps=5, warmup_steps=1, save_every=5, checkpoint_dir=args.checkpoint_dir)
    else:
        if not args.data_path or not os.path.exists(args.data_path):
            raise FileNotFoundError(f"SFT data path not found: {args.data_path}")
        needed_samples = int(args.max_steps * args.batch_size * 1.3)
        dialogues = load_dialogues_from_path(args.data_path, max_samples=needed_samples)
        cfg = getattr(OutMindConfig, args.preset)(vocab_size=tokenizer.vocab_size)
        dataset = SFTDataset(dialogues, tokenizer, max_seq_len=cfg.max_seq_len)
        trainer_cfg = TrainerConfig(max_steps=args.max_steps, checkpoint_dir=args.checkpoint_dir)

    model = OutMindForCausalLM(cfg)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state_dict = ckpt.get("model_state") or ckpt.get("model_state_dict") or ckpt
        model.load_state_dict(state_dict, strict=False)
        print(f"[OutMind] Loaded pretrain base weights from {args.checkpoint}")

    if args.use_lora:
        adapters = apply_lora(model, rank=8, alpha=16.0)
        print(f"[OutMind] LoRA activated: injected {len(adapters)} low-rank adapters.")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    trainer = Trainer(model, trainer_cfg)

    print(f"[OutMind] Starting SFT ({len(dataset)} conversation samples)...")
    trainer.train(dataloader)
    print("[OutMind] SFT completed successfully.")
    if args.output_model:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_model)), exist_ok=True)
        trainer.save_checkpoint(args.max_steps, path=args.output_model)
        print(f"[OutMind] Saved final SFT checkpoint to {args.output_model}")


if __name__ == "__main__":
    main()

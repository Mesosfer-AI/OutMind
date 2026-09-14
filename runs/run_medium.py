"""One-command execution script for OutMind-Medium (~246M parameters).

Optimized for high-performance workstation GPUs (RTX 3080/4080, RTX 3090, A10G, V100).
Runs the full end-to-end lifecycle:
  Download -> Train Tokenizer -> Eval Tokenizer -> Pretrain -> Eval Pretrain -> SFT -> Eval SFT
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from runs.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run OutMind-Medium (~246M) Pipeline")
    parser.add_argument("--mode", type=str, choices=["quick", "full"], default="quick",
                        help="'quick' runs a rapid verification cycle; 'full' executes full pretraining and SFT")
    parser.add_argument("--pretrain-steps", type=int, default=None, help="Override pretrain steps")
    parser.add_argument("--sft-steps", type=int, default=None, help="Override SFT steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cuda', or 'cpu'")
    parser.add_argument("--force-download", action="store_true", help="Redownload dataset from Hugging Face")
    parser.add_argument("--retrain-tokenizer", action="store_true", help="Retrain BPE tokenizer from scratch")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True, help="Enable activation checkpointing")
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing", help="Disable activation checkpointing")
    args = parser.parse_args()

    # Optimized defaults for Medium (Micro-batch 2 + Grad Accum 4 = Effective Batch 8)
    default_pretrain = 10 if args.mode == "quick" else 5000
    default_sft = 5 if args.mode == "quick" else 2000
    default_bs = 2
    default_grad_accum = 1 if args.mode == "quick" else 4

    run_pipeline(
        preset="medium",
        mode=args.mode,
        pretrain_steps=args.pretrain_steps or default_pretrain,
        sft_steps=args.sft_steps or default_sft,
        batch_size=args.batch_size or default_bs,
        gradient_accumulation_steps=args.gradient_accumulation_steps or default_grad_accum,
        gradient_checkpointing=args.gradient_checkpointing,
        device=args.device,
        force_download=args.force_download,
        skip_tokenizer_train=not args.retrain_tokenizer,
    )


if __name__ == "__main__":
    main()

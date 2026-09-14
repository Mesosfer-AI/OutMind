"""One-command execution script for OutMind-Large (~693M parameters).

Optimized for flagship SLM training on 24GB+ GPUs (RTX 3090/4090, A100, H100).
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
    parser = argparse.ArgumentParser(description="Run OutMind-Large (~693M) Pipeline")
    parser.add_argument("--mode", type=str, choices=["quick", "full"], default="quick",
                        help="'quick' runs a rapid verification cycle; 'full' executes full pretraining and SFT")
    parser.add_argument("--pretrain-steps", type=int, default=None, help="Override pretrain steps")
    parser.add_argument("--sft-steps", type=int, default=None, help="Override SFT steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cuda', or 'cpu'")
    parser.add_argument("--force-download", action="store_true", help="Redownload dataset from Hugging Face")
    parser.add_argument("--retrain-tokenizer", action="store_true", help="Retrain BPE tokenizer from scratch")
    args = parser.parse_args()

    # Optimized defaults for Large
    default_pretrain = 5 if args.mode == "quick" else 10000
    default_sft = 3 if args.mode == "quick" else 3000
    default_bs = 1 if args.mode == "quick" else 8

    run_pipeline(
        preset="large",
        mode=args.mode,
        pretrain_steps=args.pretrain_steps or default_pretrain,
        sft_steps=args.sft_steps or default_sft,
        batch_size=args.batch_size or default_bs,
        device=args.device,
        force_download=args.force_download,
        skip_tokenizer_train=not args.retrain_tokenizer,
    )


if __name__ == "__main__":
    main()

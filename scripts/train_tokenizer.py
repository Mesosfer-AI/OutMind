"""CLI script to train a custom BPE tokenizer on raw text files."""

import argparse
import base64
import os
from typing import Optional

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from outmind.tokenizer import INDONESIAN_BPE_PAT, SPECIAL_TOKEN_NAMES


def _bytes_to_unicode_map():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def text_iterator_from_parquet(parquet_dir: str, max_samples: int = 100000):
    import pyarrow.parquet as pq
    parquet_files = []
    for root, _, files in os.walk(parquet_dir):
        for f in files:
            if f.endswith(".parquet"):
                parquet_files.append(os.path.join(root, f))
    parquet_files.sort()
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    
    samples_per_file = max(1, max_samples // max(len(parquet_files), 1))
    total = 0
    print(f"[OutMind] Streaming from {len(parquet_files)} parquet files (target {max_samples:,} samples)...", flush=True)
    for pf in parquet_files:
        pfile = pq.ParquetFile(pf)
        count = 0
        for batch in pfile.iter_batches(batch_size=1000, columns=["text"]):
            for t in batch["text"].to_pylist():
                if t and len(t) > 50:
                    yield t
                    count += 1
                    total += 1
                    if count >= samples_per_file or total >= max_samples:
                        break
            if count >= samples_per_file or total >= max_samples:
                break
        if total >= max_samples:
            break


def train_bpe(
    data_file: Optional[str] = None,
    parquet_dir: Optional[str] = None,
    output_prefix: str = "data/processed/outmind_tokenizer",
    vocab_size: int = 64000,
    max_samples: int = 100000,
):
    source_desc = f"parquet directory {parquet_dir}" if parquet_dir else f"file {data_file}"
    print(f"[OutMind] Training BPE tokenizer from {source_desc} (target vocab: {vocab_size:,})...", flush=True)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(INDONESIAN_BPE_PAT), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteFallback()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKEN_NAMES,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    if parquet_dir:
        tokenizer.train_from_iterator(
            text_iterator_from_parquet(parquet_dir, max_samples),
            trainer,
            length=max_samples,
        )
    else:
        tokenizer.train([data_file], trainer)

    json_path = f"{output_prefix}.json"
    tokenizer.save(json_path)

    # Export contiguous tiktoken.model format (base64 token_bytes and rank)
    u2b = _bytes_to_unicode_map()
    vocab = tokenizer.get_vocab()
    regular_tokens = [
        tok for tok, _ in sorted(vocab.items(), key=lambda x: x[1])
        if tok not in SPECIAL_TOKEN_NAMES
    ]

    model_path = f"{output_prefix}.model"
    with open(model_path, "w", encoding="utf-8") as f:
        for rank, tok in enumerate(regular_tokens):
            raw_bytes = bytes([u2b[c] for c in tok])
            b64 = base64.b64encode(raw_bytes).decode("utf-8")
            f.write(f"{b64} {rank}\n")

    print(f"[OutMind] Tokenizer saved to {json_path} and {model_path} (tiktoken format)")


def main():
    parser = argparse.ArgumentParser(description="Train OutMind BPE Tokenizer")
    parser.add_argument("--data-file", type=str, default=None, help="Input plain text file")
    parser.add_argument("--parquet-dir", type=str, default=None, help="Input cleaned Parquet directory (e.g. data/cleaned/pretrain)")
    parser.add_argument("--max-samples", type=int, default=100000, help="Maximum text samples to sample across Parquet files")
    parser.add_argument("--output-prefix", type=str, default="data/processed/outmind_tokenizer")
    parser.add_argument("--vocab-size", type=int, default=64000)
    args = parser.parse_args()

    if not args.data_file and not args.parquet_dir:
        raise ValueError("Must provide either --data-file or --parquet-dir")
    if args.data_file and not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}")
    if args.parquet_dir and not os.path.exists(args.parquet_dir):
        raise FileNotFoundError(f"Parquet directory not found: {args.parquet_dir}")

    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
    train_bpe(
        data_file=args.data_file,
        parquet_dir=args.parquet_dir,
        output_prefix=args.output_prefix,
        vocab_size=args.vocab_size,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

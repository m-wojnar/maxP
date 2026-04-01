#!/usr/bin/env python3
"""Download and tokenize a subset of OpenWebText for GPT-2 training.

Uses HuggingFace `datasets` to stream OpenWebText and `tiktoken` for
GPT-2 BPE tokenization (vocab size 50257). Writes train.bin + val.bin
as np.uint16 memory-mapped files.

Usage:
    python prepare.py                     # default 100k docs (~1% of full)
    python prepare.py --num-docs 50000    # smaller subset
    python prepare.py --output-dir ./data/openwebtext
"""

import argparse
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Download and tokenize OpenWebText subset for GPT-2 training"
    )
    parser.add_argument("--num-docs", type=int, default=100_000,
                        help="Number of documents to download (default: 100k, ~1%% of full)")
    parser.add_argument("--output-dir", type=str, default="./data/openwebtext",
                        help="Output directory for train.bin and val.bin")
    parser.add_argument("--val-fraction", type=float, default=0.005,
                        help="Fraction of tokens for validation (default: 0.5%%)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]

    # Stream dataset
    print(f"Loading OpenWebText ({args.num_docs} docs)...")
    ds = load_dataset("openwebtext", split="train", streaming=True)

    # Tokenize
    all_tokens = []
    for i, example in enumerate(tqdm(ds, total=args.num_docs, desc="Tokenizing")):
        if i >= args.num_docs:
            break
        tokens = enc.encode_ordinary(example["text"])
        tokens.append(eot)
        all_tokens.extend(tokens)

    all_tokens = np.array(all_tokens, dtype=np.uint16)
    print(f"Total tokens: {len(all_tokens):,}")

    # Split into train/val
    n_val = int(len(all_tokens) * args.val_fraction)
    n_train = len(all_tokens) - n_val

    train_tokens = all_tokens[:n_train]
    val_tokens = all_tokens[n_train:]

    # Write memory-mapped files
    train_path = os.path.join(args.output_dir, "train.bin")
    val_path = os.path.join(args.output_dir, "val.bin")

    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)

    print(f"Train: {n_train:,} tokens → {train_path}")
    print(f"Val:   {n_val:,} tokens → {val_path}")
    print("Done.")


if __name__ == "__main__":
    main()

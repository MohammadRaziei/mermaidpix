"""
train_tokenizer.py — Step 2 (run via `make tokenizer`).

Trains a small byte-level BPE tokenizer directly on the Mermaid source
strings we generated (data/reconstructor/manifest.jsonl). This is one of
the two things that keeps the reconstructor small (the other being the
ViT-Tiny encoder): reusing a general-purpose 50k-token vocabulary (like
BART's, which Donut uses) means carrying a lot of embedding-table weight
for tokens this narrow domain will basically never use. A vocabulary
trained on our own corpus needs far fewer tokens to cover the same text.

Byte-level BPE (not word-level) is used specifically so any real-world
diagram's arbitrary node/label text -- not just what our synthetic
generator happened to produce -- can still be encoded, just possibly as
more sub-word tokens.

Usage:
    python train_tokenizer.py --manifest ./data/reconstructor/manifest.jsonl --out ./tokenizer
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="./data/reconstructor/manifest.jsonl")
    ap.add_argument("--vocab-size", type=int, default=6000)
    ap.add_argument("--out", type=str, default="./tokenizer")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}. Run `make data` first.")

    corpus_path = Path(tempfile.gettempdir()) / "mermaid_corpus.txt"
    n = 0
    with open(manifest_path) as f, open(corpus_path, "w") as out:
        for line in f:
            row = json.loads(line)
            out.write(row["target"] + "\n")
            n += 1
    print(f"Wrote {n} Mermaid source samples to corpus.")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(corpus_path)],
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(str(out_dir))

    # sanity check: round-trip a real sample and report compression stats
    with open(manifest_path) as f:
        sample = json.loads(f.readline())["target"]
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded.ids)

    print(f"\nTokenizer saved to {out_dir} (vocab_size={tokenizer.get_vocab_size()})")
    print(f"Sample round-trip check:")
    print(f"  original chars: {len(sample)}, tokens: {len(encoded.ids)} "
          f"({len(sample) / max(len(encoded.ids), 1):.1f} chars/token)")
    print(f"  round-trip matches exactly: {decoded.strip() == sample.strip()}")
    print(
        "\n>>> Please paste back the vocab_size, chars/token ratio, and whether "
        "the round-trip matched -- if it doesn't match exactly, something about "
        "the Mermaid syntax (quotes, special characters) needs a tokenizer tweak "
        "before we train the reconstructor on top of it."
    )


if __name__ == "__main__":
    main()
    
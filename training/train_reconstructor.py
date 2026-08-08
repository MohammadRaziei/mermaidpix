"""
train_reconstructor.py — Step 4 (run via `make train-reconstructor`).

Trains the single shared image -> Mermaid code model (see model.py for the
architecture). One model handles every diagram type -- there's no
per-type detector or hand-written assembly logic here, unlike an
object-detection approach; the decoder just learns to emit the right
Mermaid syntax directly, conditioned on what it sees.

Writes per-epoch loss/token-accuracy, a handful of qualitative
prediction-vs-ground-truth samples, and the checkpoint into
results/reconstructor/ (picked up by `make package`).

Usage:
    python train_reconstructor.py --data ./data/reconstructor --tokenizer ./tokenizer --epochs 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor

from model import MermaidReconstructor

TROCR_CHECKPOINT = "microsoft/trocr-base-stage1"
# Pulled from the actual TrOCR processor rather than hand-picking a resize/
# normalization -- this must match what the pretrained encoder expects, or
# the "pretrained" weights are being fed out-of-distribution inputs.
_processor = AutoImageProcessor.from_pretrained(TROCR_CHECKPOINT)
raw_size = _processor.size
if hasattr(raw_size, "height"):
    IMG_SIZE = int(raw_size.height)
elif hasattr(raw_size, "get"):
    IMG_SIZE = int(raw_size.get("height", raw_size.get("shortest_edge", 384)))
else:
    IMG_SIZE = int(raw_size)

IMAGE_MEAN = _processor.image_mean
IMAGE_STD = _processor.image_std


def pad_to_square(img: Image.Image, fill=255) -> Image.Image:
    """Pad (don't stretch) to square before resizing, so text aspect ratio
    -- important for OCR-like reading -- isn't distorted."""
    w, h = img.size
    side = max(w, h)
    return ImageOps.pad(img, (side, side), color=(fill, fill, fill), centering=(0.5, 0.5))


IMG_TRANSFORM = transforms.Compose([
    transforms.Lambda(lambda img: pad_to_square(img.convert("RGB"))),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
])


class ReconstructorDataset(Dataset):
    def __init__(self, root: Path, split: str, tokenizer: ByteLevelBPETokenizer, max_len: int = 640):
        self.root = root
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.bos_id = tokenizer.token_to_id("<s>")
        self.eos_id = tokenizer.token_to_id("</s>")
        self.pad_id = tokenizer.token_to_id("<pad>")

        self.rows = []
        with open(root / "manifest.jsonl") as f:
            for line in f:
                row = json.loads(line)
                if row["split"] == split:
                    self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(self.root / "images" / row["file_name"])
        img_tensor = IMG_TRANSFORM(img)

        ids = self.tokenizer.encode(row["target"]).ids
        ids = [self.bos_id] + ids[: self.max_len - 2] + [self.eos_id]
        return img_tensor, torch.tensor(ids, dtype=torch.long)


class PadCollate:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id
    
    def __call__(self, batch):
        images, id_seqs = zip(*batch)
        images = torch.stack(images)
        max_len = max(len(ids) for ids in id_seqs)
        padded = torch.full((len(id_seqs), max_len), self.pad_id, dtype=torch.long)
        for i, ids in enumerate(id_seqs):
            padded[i, : len(ids)] = ids
        return images, padded

def load_tokenizer(tokenizer_dir: Path) -> ByteLevelBPETokenizer:
    return ByteLevelBPETokenizer(
        str(tokenizer_dir / "vocab.json"),
        str(tokenizer_dir / "merges.txt"),
    )


@torch.no_grad()
def evaluate(model, loader, device, pad_id, criterion):
    model.eval()
    total_loss, total_tokens, correct_tokens = 0.0, 0, 0
    for images, ids in loader:
        images, ids = images.to(device), ids.to(device)
        decoder_input, labels = ids[:, :-1], ids[:, 1:]
        logits = model(images, decoder_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        mask = labels != pad_id
        total_loss += loss.item() * mask.sum().item()
        preds = logits.argmax(dim=-1)
        correct_tokens += ((preds == labels) & mask).sum().item()
        total_tokens += mask.sum().item()
    return total_loss / max(total_tokens, 1), correct_tokens / max(total_tokens, 1)


@torch.no_grad()
def qualitative_samples(model, dataset, tokenizer, device, n_per_type=2):
    """Samples n_per_type examples from EACH diagram type present in the val
    split, not just the first n rows overall -- the manifest is built by
    iterating diagram types in order, so val_ds[:n] was silently always
    100% flowchart examples in the previous version of this function,
    leaving every other diagram type completely unchecked."""
    by_type: dict[str, list[int]] = {}
    for idx, row in enumerate(dataset.rows):
        by_type.setdefault(row["diagram_type"], []).append(idx)

    samples = []
    for diagram_type, indices in sorted(by_type.items()):
        for idx in indices[:n_per_type]:
            img_tensor, gt_ids = dataset[idx]
            gt_text = tokenizer.decode([t for t in gt_ids.tolist() if t not in
                                         (dataset.bos_id, dataset.eos_id, dataset.pad_id)])
            pred_ids = model.generate(img_tensor.unsqueeze(0).to(device),
                                       bos_id=dataset.bos_id, eos_id=dataset.eos_id)
            pred_text = tokenizer.decode([t for t in pred_ids[0].tolist() if t not in
                                           (dataset.bos_id, dataset.eos_id, dataset.pad_id)])
            samples.append({
                "file_name": dataset.rows[idx]["file_name"],
                "diagram_type": diagram_type,
                "ground_truth": gt_text,
                "prediction": pred_text,
                "exact_match": gt_text.strip() == pred_text.strip(),
            })
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="./data/reconstructor")
    ap.add_argument("--tokenizer", type=str, default="./tokenizer")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4,
                     help="learning rate for from-scratch params (projection + decoder + head)")
    ap.add_argument("--encoder-lr", type=float, default=1e-5,
                     help="learning rate for the unfrozen (fine-tuned) encoder blocks -- kept low "
                          "so pretrained TrOCR weights aren't wrecked before the decoder catches up")
    ap.add_argument("--freeze-bottom-n-blocks", type=int, default=4,
                     help="how many of the encoder's 12 blocks to freeze entirely -- see IDEA.md")
    ap.add_argument("--results-dir", type=str, default="./results/reconstructor")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    tokenizer = load_tokenizer(Path(args.tokenizer))
    pad_id = tokenizer.token_to_id("<pad>")
    vocab_size = tokenizer.get_vocab_size()
    print(f"tokenizer vocab size: {vocab_size}")

    root = Path(args.data)
    train_ds = ReconstructorDataset(root, "train", tokenizer)
    val_ds = ReconstructorDataset(root, "val", tokenizer)
    print(f"train samples: {len(train_ds)}  val samples: {len(val_ds)}")

    collate = PadCollate(pad_id)   

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=2)

    model = MermaidReconstructor(
        vocab_size=vocab_size, pad_id=pad_id, freeze_bottom_n_blocks=args.freeze_bottom_n_blocks,
    ).to(device)
    param_counts = model.num_params()
    print("model size:", {k: f"{v/1e6:.2f}M" for k, v in param_counts.items()})
    print(f"trainable share: {param_counts['total_trainable']/param_counts['total']*100:.1f}%")

    optimizer = torch.optim.AdamW([
        {"params": model.encoder_param_groups(), "lr": args.encoder_lr},
        {"params": model.scratch_param_groups(), "lr": args.lr},
    ], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    history = []
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, ids in train_loader:
            images, ids = images.to(device), ids.to(device)
            decoder_input, labels = ids[:, :-1], ids[:, 1:]

            optimizer.zero_grad()
            logits = model(images, decoder_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, val_token_acc = evaluate(model, val_loader, device, pad_id, criterion)
        print(f"epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_token_acc={val_token_acc:.4f}")
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_token_acc": val_token_acc,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "vocab_size": vocab_size, "pad_id": pad_id},
                       results_dir / "reconstructor_model.pt")

    with open(results_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nGenerating qualitative samples on val set...")
    samples = qualitative_samples(model, val_ds, tokenizer, device, n_per_type=2)
    with open(results_dir / "qualitative_samples.json", "w") as f:
        json.dump(samples, f, indent=2)
    exact_match_rate = sum(s["exact_match"] for s in samples) / len(samples)

    summary = {
        "model_size_params": param_counts,
        "vocab_size": vocab_size,
        "best_val_loss": best_val_loss,
        "final_val_token_acc": history[-1]["val_token_acc"],
        "qualitative_exact_match_rate": exact_match_rate,
        "num_epochs": args.epochs,
        "num_train_samples": len(train_ds),
        "num_val_samples": len(val_ds),
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Final val token accuracy: {history[-1]['val_token_acc']:.4f}")
    print(f"Qualitative exact-match rate ({len(samples)} samples, ~2 per diagram type): {exact_match_rate:.2f}")
    print(f"Saved model + history + samples to {results_dir}/")
    print(
        "\n>>> Please paste back: the train/val loss curve, final val_token_acc, "
        "and a couple of the qualitative_samples.json entries (especially any "
        "exact_match=false ones) -- I'll look at *where* the prediction diverges "
        "from ground truth to tell whether it's a tokenizer issue, a specific "
        "diagram type that needs more data, or just needs more epochs."
    )


if __name__ == "__main__":
    main()

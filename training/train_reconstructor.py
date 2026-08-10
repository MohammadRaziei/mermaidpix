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
import io
import json
import random
from pathlib import Path

import mermaidx
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import Dataset, IterableDataset, DataLoader, get_worker_info
from torchvision import transforms
from transformers import AutoImageProcessor

from common.diagram_generators import (
    DIAGRAM_BUILDERS,
    random_theme_and_look,
    wrap_with_frontmatter,
)

# NOTE (see IDEA.md, "Also still open / not yet done"): this file previously
# had zero image-level augmentation, unlike train_router.py -- flagged as a
# possible contributor to the overfitting seen in the first real training
# run (val_loss got worse after ~epoch 10 while train_loss kept dropping).
# Added below. Deliberately conservative compared to train_router.py's
# RandomRotation(5): this model has to read exact label *text* and exact
# arrow *direction/topology*, not just classify a diagram type, so anything
# that meaningfully warps geometry risks teaching it the wrong structure.
# Rotation is capped much lower, and nothing here mirrors/flips the image
# (a flipped flowchart arrow reverses its real meaning).
#
# NOTE 2 -- bigger change, added in the same pass: TRAIN now defaults to
# on-the-fly generation (OnTheFlyReconstructorDataset below) instead of
# reading a fixed, pre-rendered manifest. This attacks the label-
# hallucination problem IDEA.md diagnosed at its root: with a *finite* set
# of pre-rendered images, the model can partially memorize "this exact PNG
# says X" instead of reading pixels, no matter how large the label
# vocabulary is. Rendering a fresh random diagram (new text, new node/edge
# topology, new theme/look) per sample via mermaidx -- which is pure-Python
# and browserless, so this is cheap enough to do in the data-loading path --
# makes that shortcut unavailable: no two training samples are ever the same
# image twice. VAL deliberately stays on the old fixed manifest (see
# ReconstructorDataset below) -- metrics need to be measured on the same
# held-out examples every epoch to be comparable across epochs and runs; an
# ever-changing val set would make the val_loss curve meaningless. The old
# fixed-manifest training path is kept available via --no-on-the-fly for
# direct before/after comparison, since this hasn't been validated with a
# real training run yet (see README "What's tested").

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


def add_gaussian_noise(tensor: torch.Tensor, std: float) -> torch.Tensor:
    """Applied post-ToTensor (values in [0,1]), pre-Normalize. Mimics mild
    JPEG/screenshot/rescan noise mermaidx's clean vector renders never have,
    without touching geometry (unlike rotation/crop, this can't move a label
    or an arrowhead) -- a comparatively "safe" augmentation for this task."""
    if std <= 0:
        return tensor
    return (tensor + torch.randn_like(tensor) * std).clamp(0.0, 1.0)


def build_transform(train: bool, augment_strength: float = 1.0) -> transforms.Compose:
    """augment_strength scales every augmentation op linearly; 0 disables
    augmentation entirely (equivalent to the old, pre-augmentation
    IMG_TRANSFORM) while keeping the eval-time pipeline (train=False)
    identical either way, since val/test should never be augmented."""
    ops = [transforms.Lambda(lambda img: pad_to_square(img.convert("RGB")))]

    if train and augment_strength > 0:
        ops += [
            # Small rotation only: large angles would rotate arrowheads and
            # text baselines away from what the OCR-pretrained encoder
            # expects and can flip which side of a node a label reads from.
            # fill=(255,255,255) matches pad_to_square's white background so
            # rotation doesn't introduce a visible black-corner artifact the
            # model could learn to key off of instead of real content.
            transforms.RandomRotation(2.0 * augment_strength, fill=(255, 255, 255)),
            # Brightness/contrast/saturation jitter: covers the 11 Mermaid
            # themes' color variation more broadly than the fixed set the
            # dataset already randomizes over, without ever touching pixel
            # *positions* (so labels/edges stay exactly where the target
            # text says they are).
            transforms.ColorJitter(
                brightness=0.15 * augment_strength,
                contrast=0.15 * augment_strength,
                saturation=0.10 * augment_strength,
            ),
        ]

    ops += [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]

    if train and augment_strength > 0:
        ops.append(transforms.Lambda(lambda t: add_gaussian_noise(t, std=0.02 * augment_strength)))

    ops.append(transforms.Normalize(IMAGE_MEAN, IMAGE_STD))
    return transforms.Compose(ops)


class ReconstructorDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        tokenizer: ByteLevelBPETokenizer,
        transform: transforms.Compose,
        max_len: int = 640,
    ):
        self.root = root
        self.tokenizer = tokenizer
        self.transform = transform
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
        img_tensor = self.transform(img)

        ids = self.tokenizer.encode(row["target"]).ids
        ids = [self.bos_id] + ids[: self.max_len - 2] + [self.eos_id]
        return img_tensor, torch.tensor(ids, dtype=torch.long)


class OnTheFlyReconstructorDataset(IterableDataset):
    """Renders a fresh random diagram per sample instead of reading a
    pre-rendered manifest -- see the NOTE 2 block at the top of this file
    for why. Infinite by design (the point is that no two samples repeat),
    so unlike ReconstructorDataset it has no __len__/epoch boundary; the
    caller decides how many samples make up one "epoch" via --steps-per-
    epoch and pulls that many batches from a persistent iterator (see
    main() below) rather than looping until StopIteration.
    """

    def __init__(
        self,
        tokenizer: ByteLevelBPETokenizer,
        transform: transforms.Compose,
        max_len: int = 640,
        seed: int = 0,
        render_widths: tuple[int, ...] = (600, 800, 1000),
    ):
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len
        self.seed = seed
        self.render_widths = render_widths
        self.bos_id = tokenizer.token_to_id("<s>")
        self.eos_id = tokenizer.token_to_id("</s>")
        self.pad_id = tokenizer.token_to_id("<pad>")
        self.diagram_types = list(DIAGRAM_BUILDERS.keys())

    def _make_rng(self) -> random.Random:
        # Distinct-but-deterministic seed per DataLoader worker: keeps a run
        # reproducible for a given --seed (useful for debugging a specific
        # step) while making sure num_workers>1 doesn't have every worker
        # emit the exact same stream in lockstep.
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        return random.Random(self.seed * 1_000_003 + worker_id)

    def __iter__(self):
        rng = self._make_rng()
        while True:
            diagram_type = rng.choice(self.diagram_types)
            src = DIAGRAM_BUILDERS[diagram_type](rng)
            theme, look = random_theme_and_look(rng)
            wrapped = wrap_with_frontmatter(src, theme, look)
            width = rng.choice(self.render_widths)

            try:
                diagram = mermaidx.render(wrapped)
                png_bytes = diagram.png(width=width, background="#ffffff")
            except Exception:
                # A bad random sample (e.g. a generator edge case mermaidx's
                # engine rejects) shouldn't kill the whole training run --
                # same "skip and keep going" policy generate_dataset.py uses.
                continue
            if png_bytes is None:
                continue

            img = Image.open(io.BytesIO(png_bytes))
            img_tensor = self.transform(img)

            ids = self.tokenizer.encode(src).ids
            ids = [self.bos_id] + ids[: self.max_len - 2] + [self.eos_id]
            yield img_tensor, torch.tensor(ids, dtype=torch.long)


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
    ap.add_argument("--data", type=str, default="./data/reconstructor",
                     help="fixed manifest dir -- ALWAYS used for val; also used for train "
                          "when --no-on-the-fly is passed")
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
    ap.add_argument("--augment-strength", type=float, default=1.0,
                     help="scales train-time rotation/color-jitter/noise linearly; 0 disables "
                          "augmentation. Only affects the fixed-manifest path meaningfully -- "
                          "on-the-fly generation already varies text/topology/theme per sample, "
                          "so pixel-level augmentation matters less there but is still applied "
                          "for cheap extra robustness (e.g. eventual real-world screenshots).")
    ap.add_argument("--on-the-fly", dest="on_the_fly", action="store_true", default=True,
                     help="(default) generate a fresh random diagram per training sample via "
                          "mermaidx instead of reading data/reconstructor's fixed manifest -- "
                          "see NOTE 2 near the top of this file")
    ap.add_argument("--no-on-the-fly", dest="on_the_fly", action="store_false",
                     help="use the old fixed-manifest training path (data/reconstructor must "
                          "already exist via `make data`) -- for direct before/after comparison")
    ap.add_argument("--steps-per-epoch", type=int, default=300,
                     help="only used with --on-the-fly, which has no natural epoch boundary "
                          "(the generator is infinite by design). 300 steps * batch-size 16 = "
                          "4800 fresh samples/epoch as a starting point -- raise if train_loss "
                          "looks noisy/unstable, lower if each epoch takes too long.")
    ap.add_argument("--num-workers", type=int, default=4,
                     help="DataLoader workers. With --on-the-fly this matters more than usual: "
                          "rendering happens IN the data-loading path now (CPU-bound). Measured "
                          "in-sandbox (single process, CPU): ~9s one-time mermaidx/QuickJS engine "
                          "warmup per process, then ~0.1-0.95s/render steady-state (~0.3s avg) "
                          "depending on diagram type/complexity. Each worker pays the ~9s warmup "
                          "once, not per sample -- more workers buys roughly linear steady-state "
                          "throughput after that. Size this against your GPU's forward+backward "
                          "time per batch: if data generation is slower, the GPU sits idle waiting "
                          "on it, and more workers (or fewer steps-per-epoch) is the fix.")
    ap.add_argument("--seed", type=int, default=0,
                     help="seeds the on-the-fly generator (per-worker-derived, see "
                          "OnTheFlyReconstructorDataset._make_rng) for reproducible debugging")
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
    train_tf = build_transform(train=True, augment_strength=args.augment_strength)
    eval_tf = build_transform(train=False)

    # Val ALWAYS comes from the fixed manifest, on-the-fly or not -- see
    # NOTE 2 at the top of this file for why a changing val set would make
    # val_loss meaningless across epochs.
    val_ds = ReconstructorDataset(root, "val", tokenizer, transform=eval_tf)
    collate = PadCollate(pad_id)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate, num_workers=2)

    if args.on_the_fly:
        print(f"train data: on-the-fly generation via mermaidx "
              f"({args.steps_per_epoch} steps/epoch, batch size {args.batch_size})")
        train_ds = OnTheFlyReconstructorDataset(tokenizer, transform=train_tf, seed=args.seed)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                   collate_fn=collate, num_workers=args.num_workers)
        train_iter = iter(train_loader)  # kept alive across epochs -- see class docstring
        steps_per_epoch = args.steps_per_epoch
    else:
        train_ds = ReconstructorDataset(root, "train", tokenizer, transform=train_tf)
        print(f"train data: fixed manifest, {len(train_ds)} samples")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   collate_fn=collate, num_workers=2)
        train_iter = None
        steps_per_epoch = len(train_loader)
    print(f"val samples: {len(val_ds)}")

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
        if args.on_the_fly:
            for _ in range(steps_per_epoch):
                images, ids = next(train_iter)
                images, ids = images.to(device), ids.to(device)
                decoder_input, labels = ids[:, :-1], ids[:, 1:]

                optimizer.zero_grad()
                logits = model(images, decoder_input)
                loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        else:
            for images, ids in train_loader:
                images, ids = images.to(device), ids.to(device)
                decoder_input, labels = ids[:, :-1], ids[:, 1:]

                optimizer.zero_grad()
                logits = model(images, decoder_input)
                loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        train_loss = running_loss / steps_per_epoch
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
        "on_the_fly": args.on_the_fly,
        "num_train_samples": (
            f"on-the-fly ({steps_per_epoch} steps/epoch x {args.epochs} epochs x "
            f"batch {args.batch_size} = {steps_per_epoch * args.epochs * args.batch_size} "
            "total, all distinct)" if args.on_the_fly else len(train_ds)
        ),
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
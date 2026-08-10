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
import multiprocessing as mp
import os
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import Dataset, DataLoader
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
# on-the-fly generation instead of reading a fixed, pre-rendered manifest.
# This attacks the label-hallucination problem IDEA.md diagnosed at its
# root: with a *finite* set of pre-rendered images, the model can partially
# memorize "this exact PNG says X" instead of reading pixels, no matter how
# large the label vocabulary is. Rendering a fresh random diagram (new
# text, new node/edge topology, new theme/look) per sample via mermaidx
# makes that shortcut unavailable. VAL deliberately stays on the old fixed
# manifest (see ReconstructorDataset below) -- metrics need to be measured
# on the same held-out examples every epoch to be comparable across epochs
# and runs. The old fixed-manifest training path is kept available via
# --no-on-the-fly for direct before/after comparison.
#
# NOTE 3 -- on-the-fly is implemented as a disk-backed spool queue
# (SpoolQueue below), not a plain DataLoader(num_workers=N). Requested in
# conversation specifically so a *separate, independent* process can watch
# what's actually queued for training without touching the trainer at all:
# N producer OS processes each independently render diagrams and write them
# as (png, json) file pairs into a small directory; this process reads and
# immediately deletes each pair the moment it consumes it. That directory's
# current contents ARE the live queue -- its depth is just a file count,
# and a read-only inspector (see inspect_on_the_fly.py) can list/open those
# files at any time, live, with zero IPC into the trainer. This trades a
# small amount of disk I/O (write+read a PNG instead of passing a tensor
# through DataLoader's internal shared-memory queue) for that transparency.

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


def _atomic_write(path: Path, data: bytes) -> None:
    """Write-then-rename so a reader (the trainer, or an independent
    inspector process) can never open a half-written file: os.replace is
    atomic on the same filesystem, so `path` only ever appears once it's
    fully written."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _spool_producer_loop(
    spool_dir: Path,
    seed: int,
    worker_id: int,
    max_queue_samples: int,
    render_widths: tuple[int, ...],
    poll_interval: float,
    stop_event,
) -> None:
    """Runs in its own OS process (spawned by SpoolQueue.start). Endlessly
    renders fresh random diagrams and writes each as a (png, json) file
    pair into `spool_dir` -- that directory IS the queue: a file existing
    there means "queued, not yet consumed"; the trainer deletes a pair the
    moment it reads it (see SpoolQueue.get_batch). A separate, read-only
    process can inspect exactly what's queued, right now, just by listing
    this directory -- see inspect_on_the_fly.py.

    Backpressure: if the directory already holds >= max_queue_samples
    pending pairs, this just polls and waits instead of rendering more, so
    a slow GPU step can't make disk usage grow without bound.
    """
    # Imported HERE, not at module top-level: this must be the first thing
    # that touches mermaidx in this process. Tested in conversation --
    # importing mermaidx in the PARENT before forking children looked at
    # first like it deadlocked every child (0 output for 30+ seconds); it
    # turned out to actually just be single-core CPU contention between
    # N processes' ~9s QuickJS warmup on a constrained sandbox, not a real
    # fork hazard -- but keeping the import local costs nothing and is the
    # safer default regardless of what other libraries end up sharing this
    # process in the future.
    import mermaidx

    rng = random.Random(seed * 1_000_003 + worker_id)
    diagram_types = list(DIAGRAM_BUILDERS.keys())
    counter = 0
    while not stop_event.is_set():
        pending = sum(1 for _ in spool_dir.glob("*.png"))
        if pending >= max_queue_samples:
            time.sleep(poll_interval)
            continue

        diagram_type = rng.choice(diagram_types)
        src = DIAGRAM_BUILDERS[diagram_type](rng)
        theme, look = random_theme_and_look(rng)
        wrapped = wrap_with_frontmatter(src, theme, look)
        width = rng.choice(render_widths)

        try:
            diagram = mermaidx.render(wrapped)
            png_bytes = diagram.png(width=width, background="#ffffff")
        except Exception:
            # A bad random sample shouldn't kill the producer -- same
            # "skip and keep going" policy generate_dataset.py uses.
            continue
        if png_bytes is None:
            continue

        counter += 1
        base = f"w{worker_id:02d}_{counter:08d}_{time.time_ns()}"
        meta = {
            "diagram_type": diagram_type, "theme": theme, "look": look,
            "width": width, "target": src, "worker_id": worker_id,
            "written_at": time.time(),
        }
        try:
            _atomic_write(spool_dir / f"{base}.png", png_bytes)
            _atomic_write(spool_dir / f"{base}.json", json.dumps(meta).encode("utf-8"))
        except OSError:
            # e.g. disk hiccup -- drop this one sample, keep the producer alive
            continue


class SpoolQueue:
    """Disk-backed producer/consumer queue for on-the-fly training data --
    see NOTE 3 near the top of this file for the reasoning. `num_workers`
    producer processes fill `spool_dir`; get_batch() (called from the main
    training process) reads and deletes files as it consumes them.
    """

    def __init__(
        self,
        spool_dir: Path,
        tokenizer: ByteLevelBPETokenizer,
        transform: transforms.Compose,
        num_workers: int,
        queue_depth_batches: int,
        batch_size: int,
        seed: int = 0,
        max_len: int = 640,
        poll_interval: float = 0.05,
        render_widths: tuple[int, ...] = (600, 800, 1000),
    ):
        self.spool_dir = spool_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len
        self.poll_interval = poll_interval
        self.bos_id = tokenizer.token_to_id("<s>")
        self.eos_id = tokenizer.token_to_id("</s>")
        self.pad_id = tokenizer.token_to_id("<pad>")
        self.max_queue_samples = queue_depth_batches * batch_size

        # Fresh start: don't let leftover files from a previous (e.g.
        # crashed) run get silently consumed as if they were live data.
        if spool_dir.exists():
            shutil.rmtree(spool_dir)
        spool_dir.mkdir(parents=True, exist_ok=True)

        self._stop_event = mp.Event()
        self._procs = [
            mp.Process(
                target=_spool_producer_loop,
                args=(spool_dir, seed, i, self.max_queue_samples,
                      render_widths, poll_interval, self._stop_event),
                daemon=True,
            )
            for i in range(num_workers)
        ]
        for p in self._procs:
            p.start()

    def qsize(self) -> int:
        """Current queue depth in samples -- what an independent inspector
        (or this process's own logging) sees by listing spool_dir."""
        return sum(1 for _ in self.spool_dir.glob("*.png"))

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        images: list[torch.Tensor] = []
        id_lists: list[torch.Tensor] = []
        while len(images) < batch_size:
            for png_path in sorted(self.spool_dir.glob("*.png")):
                if len(images) >= batch_size:
                    break
                json_path = png_path.with_suffix(".json")
                try:
                    with open(json_path) as f:
                        meta = json.load(f)
                    img = Image.open(png_path)
                    img.load()  # force full read into memory before we delete the file
                except (FileNotFoundError, OSError, json.JSONDecodeError):
                    # extremely unlikely (only this process deletes), but a
                    # torn read shouldn't crash training -- just skip it
                    continue
                finally:
                    png_path.unlink(missing_ok=True)
                    json_path.unlink(missing_ok=True)

                img_tensor = self.transform(img.convert("RGB"))
                ids = self.tokenizer.encode(meta["target"]).ids
                ids = [self.bos_id] + ids[: self.max_len - 2] + [self.eos_id]
                images.append(img_tensor)
                id_lists.append(torch.tensor(ids, dtype=torch.long))

            if len(images) < batch_size:
                time.sleep(self.poll_interval)

        max_len = max(len(ids) for ids in id_lists)
        padded = torch.full((len(id_lists), max_len), self.pad_id, dtype=torch.long)
        for i, ids in enumerate(id_lists):
            padded[i, : len(ids)] = ids
        return torch.stack(images), padded

    def shutdown(self) -> None:
        self._stop_event.set()
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        shutil.rmtree(self.spool_dir, ignore_errors=True)


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
                          "(production is infinite by design). 300 steps * batch-size 16 = "
                          "4800 fresh samples/epoch as a starting point -- raise if train_loss "
                          "looks noisy/unstable, lower if each epoch takes too long.")
    ap.add_argument("--num-workers", type=int, default=4,
                     help="number of producer OS processes rendering diagrams in parallel for "
                          "--on-the-fly (see SpoolQueue). Measured in-sandbox (single process, "
                          "CPU): ~9s one-time mermaidx/QuickJS engine warmup per process, then "
                          "~0.1-0.95s/render steady-state (~0.3s avg) depending on diagram type. "
                          "Each worker pays the ~9s warmup once, not per sample. Size this against "
                          "your GPU's forward+backward time per batch: if production is slower, "
                          "the trainer blocks in get_batch() waiting on the queue -- watch the "
                          "'queue depth' logged each epoch and add workers if it's often near 0."
                          "With --no-on-the-fly this is instead passed straight through as "
                          "DataLoader(num_workers=...) for the fixed-manifest path.")
    ap.add_argument("--queue-depth-batches", type=int, default=10,
                     help="only used with --on-the-fly. Producers stop rendering once the spool "
                          "queue holds this many batches' worth of samples, so a slow GPU step "
                          "can't make disk usage grow without bound. Also caps how much can ever "
                          "be 'in flight' for inspect_on_the_fly.py to look at.")
    ap.add_argument("--seed", type=int, default=0,
                     help="seeds the on-the-fly generator (per-worker-derived, see "
                          "_spool_producer_loop) for reproducible debugging")
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

    train_queue = None  # SpoolQueue, only used when args.on_the_fly
    if args.on_the_fly:
        spool_dir = results_dir / "otf_queue"
        print(f"train data: on-the-fly generation via {args.num_workers} producer processes, "
              f"disk-spool queue at {spool_dir} (depth cap {args.queue_depth_batches} batches, "
              f"{args.steps_per_epoch} steps/epoch, batch size {args.batch_size})")
        print(f"  -> while training runs, inspect the live queue with: "
              f"python inspect_on_the_fly.py --debug-dir {spool_dir}")
        print(f"  -> queue depth will read 0 for the first ~10-15s: each of the "
              f"{args.num_workers} producer processes pays a one-time mermaidx/QuickJS "
              f"engine warmup before its first render (measured ~9s on an otherwise-idle "
              f"CPU core in testing -- longer if workers > CPU cores, since they then "
              f"compete for the same core during that CPU-bound warmup). This is normal, "
              f"not a hang.")
        train_queue = SpoolQueue(
            spool_dir, tokenizer, train_tf,
            num_workers=args.num_workers,
            queue_depth_batches=args.queue_depth_batches,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        steps_per_epoch = args.steps_per_epoch
    else:
        train_ds = ReconstructorDataset(root, "train", tokenizer, transform=train_tf)
        print(f"train data: fixed manifest, {len(train_ds)} samples")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   collate_fn=collate, num_workers=2)
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
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            if args.on_the_fly:
                for _ in range(steps_per_epoch):
                    images, ids = train_queue.get_batch(args.batch_size)
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
            queue_note = f"  queue_depth={train_queue.qsize()}" if args.on_the_fly else ""
            print(f"epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_token_acc={val_token_acc:.4f}{queue_note}")
            history.append({
                "epoch": epoch, "train_loss": train_loss,
                "val_loss": val_loss, "val_token_acc": val_token_acc,
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"model_state": model.state_dict(), "vocab_size": vocab_size, "pad_id": pad_id},
                           results_dir / "reconstructor_model.pt")
    finally:
        # Always stop the producer processes and clean up the spool
        # directory, even on Ctrl-C or an exception mid-epoch -- otherwise
        # mermaidx-rendering worker processes are left running in the
        # background after the script exits.
        if train_queue is not None:
            train_queue.shutdown()

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

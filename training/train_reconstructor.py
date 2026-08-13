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
import logging
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor

from hf_offline_first import from_pretrained_offline_first

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
_processor = from_pretrained_offline_first(AutoImageProcessor, TROCR_CHECKPOINT)
raw_size = _processor.size
if hasattr(raw_size, "height"):
    IMG_SIZE = int(raw_size.height)
elif hasattr(raw_size, "get"):
    IMG_SIZE = int(raw_size.get("height", raw_size.get("shortest_edge", 384)))
else:
    IMG_SIZE = int(raw_size)

IMAGE_MEAN = _processor.image_mean
IMAGE_STD = _processor.image_std


def pad_to_square(img: Image.Image, fill=255, centering: tuple[float, float] = (0.5, 0.5)) -> Image.Image:
    """Pad (don't stretch) to square before resizing, so text aspect ratio
    -- important for OCR-like reading -- isn't distorted. `centering`
    controls where the original image sits within the padded square:
    (0.5, 0.5) is centered (mermaidx's own renders always come out
    perfectly centered in a tight bounding box -- see build_transform's
    train-time override for why that's worth varying)."""
    w, h = img.size
    side = max(w, h)
    return ImageOps.pad(img, (side, side), color=(fill, fill, fill), centering=centering)


def add_gaussian_noise(tensor: torch.Tensor, std: float) -> torch.Tensor:
    """Applied post-ToTensor (values in [0,1]), pre-Normalize. Mimics mild
    JPEG/screenshot/rescan noise mermaidx's clean vector renders never have,
    without touching geometry (unlike rotation/crop, this can't move a label
    or an arrowhead) -- a comparatively "safe" augmentation for this task."""
    if std <= 0:
        return tensor
    return (tensor + torch.randn_like(tensor) * std).clamp(0.0, 1.0)


# The three transform "ops" below are plain classes with __call__, not
# closures/lambdas, on purpose: DataLoader(num_workers>0) has to pickle the
# whole Dataset (transform included) to hand it to worker processes. On
# Linux this often goes unnoticed because the default 'fork' start method
# doesn't need pickling at all -- but Windows (and macOS with 'spawn') always
# does, and local lambdas/closures aren't picklable, which surfaces as
# "Can't pickle local object 'build_transform.<locals>.<lambda>'" the moment
# --no-on-the-fly's train_loader or (either mode's) val_loader tries to start
# its num_workers>0 worker processes. A class defined at module level with no
# unpicklable state (just plain floats/tuples in __init__) pickles fine
# everywhere. Confirmed as the actual cause of this failure on a real Windows
# run in conversation.
class _EvalPad:
    """Eval-time padding: fixed, centered -- the deterministic pipeline
    that must be identical every time (see build_transform)."""

    def __call__(self, img: Image.Image) -> Image.Image:
        return pad_to_square(img.convert("RGB"))


class _RandomAsymmetricPad:
    """Train-time padding with randomized (asymmetric) centering -- see
    build_transform's comment on why this augmentation exists."""

    def __init__(self, low: float = 0.15, high: float = 0.85):
        self.low = low
        self.high = high

    def __call__(self, img: Image.Image) -> Image.Image:
        return pad_to_square(
            img.convert("RGB"),
            centering=(random.uniform(self.low, self.high), random.uniform(self.low, self.high)),
        )


class _AddGaussianNoise:
    """Train-time tensor noise -- see add_gaussian_noise's docstring."""

    def __init__(self, std: float):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return add_gaussian_noise(tensor, self.std)


def build_transform(train: bool, augment_strength: float = 1.0) -> transforms.Compose:
    """augment_strength scales every augmentation op linearly; 0 disables
    augmentation entirely (equivalent to the old, pre-augmentation
    IMG_TRANSFORM) while keeping the eval-time pipeline (train=False)
    identical either way, since val/test should never be augmented."""
    if train and augment_strength > 0:
        # Asymmetric padding: mermaidx always renders its diagram tightly
        # and perfectly centered in the frame (pad_to_square's default
        # centering=(0.5, 0.5)). Every real-world screenshot or photo this
        # model might see at actual inference time won't be that clean --
        # uneven margins, the diagram pushed toward one edge, etc. Without
        # this, the model could learn to *rely on* perfect centering as a
        # signal (e.g. "the content always starts at the same relative
        # position") in a way that doesn't hold outside of mermaidx's own
        # output. Range kept away from the 0/1 extremes so the diagram is
        # never at risk of being clipped by the padding box.
        pad_op = _RandomAsymmetricPad()
    else:
        pad_op = _EvalPad()
    ops = [pad_op]

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
        ops.append(_AddGaussianNoise(std=0.02 * augment_strength))

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
    render_engines: tuple[str, ...] = ("quickjs",),
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

    render_engines: which backend(s) to render each sample with, chosen
    uniformly at random per sample. "quickjs" is mermaidx's own default
    engine; "merman" and "mermaid-rs-renderer" (via the optional `mmdr`
    package -- both independent Rust reimplementations, no JS/mermaid.js
    involved) add real visual diversity for free, since they're still
    100% labeled (same source text fed to every engine). Verified in
    conversation: merman renders near-pixel-identical to mermaidx's own
    QuickJS output (same colors/layout), while mermaid-rs-renderer differs
    meaningfully -- different color scheme, different arrow style, and its
    own layout choices can even mirror which side a branch appears on
    (e.g. a Yes/No decision's left/right placement can flip relative to
    quickjs/merman for the identical input text). That's fine, not a
    labeling bug: the target text is renderer-agnostic by construction
    (same source text regardless of which engine drew it), so this is
    exactly the kind of visual-style diversity augmentation is supposed to
    provide, the same way theme/color-jitter already does -- the engine
    used is recorded in each sample's metadata for later analysis, but is
    deliberately NOT fed to the model as an input (see IDEA.md: unlike
    diagram_type, "which renderer produced this pixel image" isn't a
    property a real-world image would ever have, so conditioning on it
    wouldn't generalize past this synthetic dataset).
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

    non_quickjs_engines = [e for e in render_engines if e != "quickjs"]
    mmdr = None
    if non_quickjs_engines:
        try:
            import mmdr as _mmdr
            mmdr = _mmdr
        except ImportError:
            print(f"[w{worker_id}] WARNING: --render-engines requested {non_quickjs_engines} "
                  f"but the optional `mmdr` package isn't installed (pip install mmdr) -- "
                  f"falling back to quickjs only for this worker.", flush=True)
            render_engines = ("quickjs",)

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
        engine = rng.choice(render_engines)

        try:
            if engine == "quickjs":
                diagram = mermaidx.render(wrapped)
            else:
                diagram = mmdr.render(wrapped, backend=engine)
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
            "engine": engine,  # provenance/debugging only -- never fed to the model, see docstring
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
        render_engines: tuple[str, ...] = ("quickjs",),
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
                      render_widths, poll_interval, self._stop_event, render_engines),
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


def setup_logging(results_dir: Path) -> logging.Logger:
    """Logs to BOTH stdout (so tmux/a live terminal shows progress -- the
    whole point being requested in conversation was "it looks frozen in
    tmux") AND results/reconstructor/train.log (so the full log travels
    inside results.zip -- package_results.py already zips everything under
    results/ recursively, so this needs no changes there). Timestamps on
    every line double as the timing info asked for: diffing two line
    timestamps tells you the actual wall-clock rate directly from the log,
    even without the explicit steps/sec numbers logged during training."""
    logger = logging.getLogger("train_reconstructor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate lines if main() ever runs twice in one process
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(results_dir / "train.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


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
    ap.add_argument("--render-engines", nargs="+", default=["quickjs"],
                     choices=["quickjs", "merman", "mermaid-rs-renderer"],
                     help="only used with --on-the-fly. Each sample is rendered with one engine "
                          "chosen uniformly at random from this list -- 'quickjs' is mermaidx's "
                          "own default; 'merman' and 'mermaid-rs-renderer' (independent Rust "
                          "reimplementations, via the optional `mmdr` package -- pip install mmdr) "
                          "add real visual diversity for free, since the target text stays "
                          "renderer-agnostic. Verified in conversation: merman renders near-"
                          "identical to quickjs, mermaid-rs-renderer differs meaningfully "
                          "(different colors/arrow style, and can even mirror left/right branch "
                          "placement) -- see the docstring on _spool_producer_loop. Default is "
                          "quickjs-only so `mmdr` stays an optional dependency; try e.g. "
                          "--render-engines quickjs merman mermaid-rs-renderer for the full mix.")
    ap.add_argument("--seed", type=int, default=0,
                     help="seeds the on-the-fly generator (per-worker-derived, see "
                          "_spool_producer_loop) for reproducible debugging")
    ap.add_argument("--results-dir", type=str, default="./results/reconstructor")
    ap.add_argument("--log-every", type=int, default=20,
                     help="log a progress line every N training steps (in addition to the "
                          "once-per-epoch summary line) -- exists specifically so a long epoch "
                          "(e.g. --steps-per-epoch 300 with slow on-the-fly rendering) doesn't "
                          "sit silent long enough to look frozen in a terminal/tmux session. Set "
                          "higher for less noise, or very high to effectively disable.")
    ap.add_argument("--qualitative-n-per-type", type=int, default=20,
                     help="how many val samples per diagram type to include in "
                          "qualitative_samples.json (was hardcoded to 2 -- too small a sample to "
                          "tell a real per-type pattern from n=2 noise, see conversation's Round 2 "
                          "analysis of results1.zip).")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(results_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    tokenizer = load_tokenizer(Path(args.tokenizer))
    pad_id = tokenizer.token_to_id("<pad>")
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"tokenizer vocab size: {vocab_size}")

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
        logger.info(f"train data: on-the-fly generation via {args.num_workers} producer processes, "
                    f"disk-spool queue at {spool_dir} (depth cap {args.queue_depth_batches} batches, "
                    f"{args.steps_per_epoch} steps/epoch, batch size {args.batch_size})")
        logger.info(f"  -> while training runs, inspect the live queue with: "
                    f"python inspect_on_the_fly.py --debug-dir {spool_dir}")
        logger.info(f"  -> queue depth will read 0 for the first ~10-15s: each of the "
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
            render_engines=tuple(args.render_engines),
        )
        steps_per_epoch = args.steps_per_epoch
    else:
        train_ds = ReconstructorDataset(root, "train", tokenizer, transform=train_tf)
        logger.info(f"train data: fixed manifest, {len(train_ds)} samples")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   collate_fn=collate, num_workers=2)
        steps_per_epoch = len(train_loader)
    logger.info(f"val samples: {len(val_ds)}")

    model = MermaidReconstructor(
        vocab_size=vocab_size, pad_id=pad_id, freeze_bottom_n_blocks=args.freeze_bottom_n_blocks,
    ).to(device)
    param_counts = model.num_params()
    logger.info(f"model size: {{{', '.join(f'{k}={v/1e6:.2f}M' for k, v in param_counts.items())}}}")
    logger.info(f"trainable share: {param_counts['total_trainable']/param_counts['total']*100:.1f}%")

    optimizer = torch.optim.AdamW([
        {"params": model.encoder_param_groups(), "lr": args.encoder_lr},
        {"params": model.scratch_param_groups(), "lr": args.lr},
    ], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    history = []
    best_val_loss = float("inf")
    train_start_time = time.time()
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            epoch_start = time.time()

            def log_step_progress(step: int) -> None:
                """Shared by both branches below -- every --log-every steps,
                print elapsed/rate/ETA so a long, quiet epoch (on-the-fly
                especially, where a step can genuinely take a second or more
                once rendering is the bottleneck -- see SpoolQueue) doesn't
                look indistinguishable from a hang in a terminal/tmux
                session with nothing else on screen to prove otherwise."""
                if step % args.log_every != 0 and step != steps_per_epoch:
                    return
                elapsed = time.time() - epoch_start
                steps_per_sec = step / elapsed if elapsed > 0 else 0.0
                samples_per_sec = steps_per_sec * args.batch_size
                eta = (steps_per_epoch - step) / steps_per_sec if steps_per_sec > 0 else float("inf")
                avg_loss = running_loss / step
                queue_note = f"  queue_depth={train_queue.qsize()}" if args.on_the_fly else ""
                logger.info(
                    f"  epoch {epoch:3d}/{args.epochs}  step {step:4d}/{steps_per_epoch}  "
                    f"loss={avg_loss:.4f}  {steps_per_sec:.2f} step/s  {samples_per_sec:.1f} "
                    f"samples/s  elapsed={format_eta(elapsed)}  eta={format_eta(eta)}{queue_note}"
                )

            if args.on_the_fly:
                for step in range(1, steps_per_epoch + 1):
                    images, ids = train_queue.get_batch(args.batch_size)
                    images, ids = images.to(device), ids.to(device)
                    decoder_input, labels = ids[:, :-1], ids[:, 1:]

                    optimizer.zero_grad()
                    logits = model(images, decoder_input)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    log_step_progress(step)
            else:
                for step, (images, ids) in enumerate(train_loader, start=1):
                    images, ids = images.to(device), ids.to(device)
                    decoder_input, labels = ids[:, :-1], ids[:, 1:]

                    optimizer.zero_grad()
                    logits = model(images, decoder_input)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    log_step_progress(step)

            train_loss = running_loss / steps_per_epoch
            val_start = time.time()
            val_loss, val_token_acc = evaluate(model, val_loader, device, pad_id, criterion)
            epoch_elapsed = time.time() - epoch_start
            total_elapsed = time.time() - train_start_time
            queue_note = f"  queue_depth={train_queue.qsize()}" if args.on_the_fly else ""
            logger.info(
                f"epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  val_token_acc={val_token_acc:.4f}{queue_note}  "
                f"epoch_time={format_eta(epoch_elapsed)} (val eval {format_eta(time.time() - val_start)})  "
                f"total_elapsed={format_eta(total_elapsed)}  "
                f"est_remaining={format_eta(epoch_elapsed * (args.epochs - epoch))}"
            )
            history.append({
                "epoch": epoch, "train_loss": train_loss,
                "val_loss": val_loss, "val_token_acc": val_token_acc,
                "epoch_seconds": round(epoch_elapsed, 1),
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"model_state": model.state_dict(), "vocab_size": vocab_size, "pad_id": pad_id},
                           results_dir / "reconstructor_model.pt")
                logger.info(f"  -> new best val_loss, checkpoint saved")
    finally:
        # Always stop the producer processes and clean up the spool
        # directory, even on Ctrl-C or an exception mid-epoch -- otherwise
        # mermaidx-rendering worker processes are left running in the
        # background after the script exits.
        if train_queue is not None:
            train_queue.shutdown()

    with open(results_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("\nGenerating qualitative samples on val set...")
    samples = qualitative_samples(model, val_ds, tokenizer, device, n_per_type=args.qualitative_n_per_type)
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
        "total_train_seconds": round(time.time() - train_start_time, 1),
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nBest val loss: {best_val_loss:.4f}")
    logger.info(f"Final val token accuracy: {history[-1]['val_token_acc']:.4f}")
    logger.info(f"Qualitative exact-match rate ({len(samples)} samples, "
                f"~{args.qualitative_n_per_type} per diagram type): {exact_match_rate:.2f}")
    logger.info(f"Total training time: {format_eta(time.time() - train_start_time)}")
    logger.info(f"Saved model + history + samples + train.log to {results_dir}/")
    logger.info(
        "\n>>> Please paste back: the train/val loss curve, final val_token_acc, "
        "and a couple of the qualitative_samples.json entries (especially any "
        "exact_match=false ones) -- I'll look at *where* the prediction diverges "
        "from ground truth to tell whether it's a tokenizer issue, a specific "
        "diagram type that needs more data, or just needs more epochs."
    )


if __name__ == "__main__":
    main()
    
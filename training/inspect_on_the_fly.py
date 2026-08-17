"""
inspect_on_the_fly.py — read-only, independent inspector for the on-the-fly
reconstructor training data (see SampleQueue / NOTE 3 in
train_reconstructor.py).

The real training queue is in-RAM (multiprocessing.Queue, not disk) as of
this version, so there's nothing on disk to inspect UNLESS training was
started with --debug-dump. When it is, producers ALSO mirror each sample
to a small, bounded, rotating set of per-worker slots on disk, purely for
this script to look at -- never read back into training. That directory
is cleared once per epoch by the trainer, so what you see here is "roughly
the last few seconds of production," not the actual pending queue depth
(there's no way to see that from outside anymore -- it's in RAM inside the
trainer process). This script only ever reads; it never deletes anything,
so it can't interfere with training even if you leave it running.

Usage (only meaningful if training was started with --debug-dump):
    # one-shot snapshot
    python inspect_on_the_fly.py --debug-dir ./results/reconstructor/otf_debug

    # keep re-checking every few seconds, like `watch`
    python inspect_on_the_fly.py --debug-dir ./results/reconstructor/otf_debug --watch

    # also copy the current snapshot's images somewhere you can open with a
    # normal image viewer
    python inspect_on_the_fly.py --debug-dir ./results/reconstructor/otf_debug --copy-to ./otf_preview
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def inspect_once(debug_dir: Path, copy_to: Path | None = None) -> None:
    if not debug_dir.exists():
        print(f"{debug_dir} does not exist yet -- has train_reconstructor.py --on-the-fly started?")
        return

    png_paths = sorted(debug_dir.glob("*.png"))
    now = time.time()
    print(f"\n[{time.strftime('%H:%M:%S')}] {len(png_paths)} samples currently in debug mirror: {debug_dir}")

    if not png_paths:
        print("  (empty -- either the trainer just consumed everything, or producers "
              "are still warming up / stalled; empty is fine if it's brief)")
        return

    if copy_to is not None:
        copy_to.mkdir(parents=True, exist_ok=True)

    ok, corrupt, missing_meta = 0, 0, 0
    by_type: dict[str, int] = {}
    for png_path in png_paths:
        json_path = png_path.with_suffix(".json")
        # Read-only: never touches/deletes the file the trainer might also
        # be about to read -- a torn read here (file deleted mid-read by
        # the trainer) is expected sometimes and just skipped, not an error.
        try:
            with open(json_path) as f:
                meta = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            missing_meta += 1
            continue

        try:
            img = Image.open(png_path)
            img.verify()  # raises if the PNG is truncated/corrupt
            w, h = img.size
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            corrupt += 1
            continue

        ok += 1
        by_type[meta.get("diagram_type", "?")] = by_type.get(meta.get("diagram_type", "?"), 0) + 1
        age_s = now - meta.get("written_at", now)
        target_preview = meta.get("target", "").replace("\n", " \\n ")[:70]
        print(f"  [{meta.get('diagram_type', '?'):<14}] {w}x{h}px  age={age_s:5.1f}s"
              f"  worker={meta.get('worker_id', '?')}  target: {target_preview}...")

        if copy_to is not None:
            shutil.copy(png_path, copy_to / png_path.name)

    print(f"  -> {ok} valid, {corrupt} corrupt/unreadable, {missing_meta} missing metadata")
    if by_type:
        print(f"  -> diagram types currently queued: {dict(sorted(by_type.items()))}")
    if corrupt > 0:
        print("  !! corrupt PNGs in the queue usually mean a truncated mermaidx render or a "
              "disk-full condition on the producer side -- worth checking producer stderr.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--debug-dir", type=str, default="./results/reconstructor/otf_debug",
                     help="the debug-mirror directory -- only populated if training was started "
                          "with --debug-dump; must match --results-dir/otf_debug from the running "
                          "the running train_reconstructor.py --on-the-fly process")
    ap.add_argument("--watch", action="store_true",
                     help="keep re-checking every --interval seconds instead of a single snapshot")
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--copy-to", type=str, default=None,
                     help="also copy every currently-queued PNG here, so you can open them in a "
                          "normal image viewer instead of just reading this script's text report")
    args = ap.parse_args()

    debug_dir = Path(args.debug_dir)
    copy_to = Path(args.copy_to) if args.copy_to else None

    if args.watch:
        print(f"Watching {debug_dir} every {args.interval}s -- Ctrl-C to stop.")
        try:
            while True:
                inspect_once(debug_dir, copy_to)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nstopped.")
    else:
        inspect_once(debug_dir, copy_to)


if __name__ == "__main__":
    main()

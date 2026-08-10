"""
generate_dataset.py — Step 1 (run via `make data`).

Renders every training image using `mermaidx` -- a pure-Python package
(QuickJS + resvg under the hood, no Node/npm/Chromium/puppeteer install
required). This replaced an earlier version that shelled out to the real
`mmdc` (mermaid-cli) binary; that version worked in principle but needed a
Chromium download via puppeteer that turned out to be a real installation
headache in practice. mermaidx also has the advantage of being directly
testable in my own sandbox (no network-restricted Chromium download
needed), so unlike the mmdc version, this one has actually been run
end-to-end here -- see README "What's tested".

Produces two datasets:

1. data/router/{train,val}/{class}/*.png
   Folder-per-class images for the diagram-type classifier. Classes:
   not_diagram, flowchart, sequence, class_diagram, er_diagram,
   state_diagram, gantt, pie, mindmap, and the rest of the 29 supported
   types (see common/diagram_generators.py).

2. data/reconstructor/{train,val}/images/*.png + manifest.jsonl
   Image -> ground-truth Mermaid source pairs, for the image-to-code model.
   The ground truth is exactly the .mmd text we generated (no OCR, no box
   parsing needed -- we already know the answer because we wrote it).
"""
from __future__ import annotations

import argparse
import io
import json
import random
import tempfile
from pathlib import Path

import mermaidx
from PIL import Image

from common.diagram_generators import (
    DIAGRAM_BUILDERS,
    ROUTER_CLASSES,
    build_random_negative,
    random_theme_and_look,
    wrap_with_frontmatter,
)


def render_png(source: str, width: int = 800, background: str = "#ffffff") -> bytes | None:
    """Renders Mermaid source to PNG bytes via mermaidx. Returns None on
    failure instead of raising, so one bad random sample doesn't kill an
    entire generation run."""
    try:
        diagram = mermaidx.render(source)
        return diagram.png(width=width, background=background)
    except Exception as e:
        print(f"  [skip] render failed: {e}")
        return None


def generate_router_dataset(out_dir: Path, n_per_class: int, seed: int, val_fraction=0.15):
    rng = random.Random(seed)
    n_val = int(n_per_class * val_fraction)

    for cls in ROUTER_CLASSES:
        for split in ["train", "val"]:
            (out_dir / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in ROUTER_CLASSES:
        ok = 0
        for i in range(n_per_class):
            split = "val" if i < n_val else "train"
            path = out_dir / split / cls / f"{cls}_{i:05d}.png"

            if cls == "not_diagram":
                build_random_negative(rng).save(path)
                ok += 1
            else:
                src = DIAGRAM_BUILDERS[cls](rng)
                theme, look = random_theme_and_look(rng)
                wrapped = wrap_with_frontmatter(src, theme, look)
                png = render_png(wrapped, width=rng.choice([600, 800, 1000]))
                if png is not None:
                    with open(path, "wb") as f:
                        f.write(png)
                    ok += 1
        print(f"  router/{cls}: {ok}/{n_per_class} rendered")


def generate_reconstructor_dataset(out_dir: Path, n_per_type: int, seed: int, val_fraction=0.15):
    """Legacy/comparison path: renders BOTH train and val images to disk,
    the way this whole file worked before on-the-fly training existed. Only
    still called for --fixed-train-n-per-type > 0 (see __main__), i.e. only
    when someone explicitly wants a `train_reconstructor.py --no-on-the-fly`
    run to compare against the default. Writes to <out>/reconstructor_fixed,
    NOT <out>/reconstructor, so it never collides with the val-only set
    generate_reconstructor_val_set() below writes for the default path."""
    rng = random.Random(seed + 1)
    n_val = int(n_per_type * val_fraction)

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    manifest = []

    for diagram_type, builder in DIAGRAM_BUILDERS.items():
        ok = 0
        for i in range(n_per_type):
            split = "val" if i < n_val else "train"
            src = builder(rng)
            theme, look = random_theme_and_look(rng)
            wrapped = wrap_with_frontmatter(src, theme, look)
            fname = f"{diagram_type}_{i:05d}.png"
            path = out_dir / "images" / fname

            png = render_png(wrapped, width=rng.choice([600, 800, 1000]))
            if png is not None:
                with open(path, "wb") as f:
                    f.write(png)
                manifest.append({
                    "file_name": fname,
                    "diagram_type": diagram_type,
                    "target": src,  # clean source (no frontmatter) -- theme/look is a
                                     # rendering style choice, not part of the diagram's
                                     # content, so the model shouldn't need to reproduce it
                    "theme": theme,
                    "look": look,
                    "split": split,
                })
                ok += 1
        print(f"  reconstructor_fixed/{diagram_type}: {ok}/{n_per_type} rendered")

    with open(out_dir / "manifest.jsonl", "w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    print(f"  manifest -> {out_dir / 'manifest.jsonl'} ({len(manifest)} rows)")


def generate_reconstructor_val_set(out_dir: Path, n_per_type_val: int, seed: int):
    """DEFAULT path's val data. train_reconstructor.py's on-the-fly mode
    (the default -- see train_reconstructor.py) generates its own train
    images live via mermaidx during training, so this only ever needs to
    render the held-out VAL split -- val has to be a small, fixed,
    reproducible set so val_loss is comparable across epochs and across
    runs, which an ever-changing on-the-fly val set could never give you.
    Writes to <out_dir> directly (default: data/reconstructor/), which is
    train_reconstructor.py's --data default."""
    rng = random.Random(seed + 1)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    manifest = []

    for diagram_type, builder in DIAGRAM_BUILDERS.items():
        ok = 0
        for i in range(n_per_type_val):
            src = builder(rng)
            theme, look = random_theme_and_look(rng)
            wrapped = wrap_with_frontmatter(src, theme, look)
            fname = f"{diagram_type}_val_{i:05d}.png"
            path = out_dir / "images" / fname

            png = render_png(wrapped, width=rng.choice([600, 800, 1000]))
            if png is not None:
                with open(path, "wb") as f:
                    f.write(png)
                manifest.append({
                    "file_name": fname,
                    "diagram_type": diagram_type,
                    "target": src,
                    "theme": theme,
                    "look": look,
                    "split": "val",  # every row here is val -- there is no train
                                      # split in this file on purpose
                })
                ok += 1
        print(f"  reconstructor-val/{diagram_type}: {ok}/{n_per_type_val} rendered")

    with open(out_dir / "manifest.jsonl", "w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    print(f"  manifest -> {out_dir / 'manifest.jsonl'} ({len(manifest)} rows, val-only)")


def generate_tokenizer_corpus(out_path: Path, n_per_type: int, seed: int):
    """Text-only Mermaid sources for train_tokenizer.py -- which only ever
    reads row["target"] (see train_tokenizer.py), never the image -- so this
    skips mermaidx.render() entirely and is roughly three orders of
    magnitude cheaper per sample than the val set above. That means it can
    afford a much larger N for better vocabulary coverage/diversity than
    what would be practical if every sample also had to be rendered."""
    rng = random.Random(seed + 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w") as f:
        for diagram_type, builder in DIAGRAM_BUILDERS.items():
            for _ in range(n_per_type):
                src = builder(rng)
                f.write(json.dumps({"diagram_type": diagram_type, "target": src}) + "\n")
                n += 1
    print(f"  tokenizer corpus -> {out_path} ({n} text-only rows, no images rendered)")


def smoke_test():
    print("Running smoke test: rendering one tiny flowchart...")
    png = render_png("flowchart TD\n    A[Start] --> B[End]")
    if png is not None:
        out_path = Path(tempfile.gettempdir()) / "mermaidx_smoketest.png"
        with open(out_path, "wb") as f:
            f.write(png)
        print(f"OK -- mermaidx is working. Wrote a test image to {out_path}")
        return True
    print("FAILED -- mermaidx did not produce an image. Check `make install` output.")
    return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="./data")
    ap.add_argument("--router-n-per-class", type=int, default=300)
    ap.add_argument("--reconstructor-val-n-per-type", type=int, default=60,
                     help="rendered images per diagram type for the reconstructor's fixed VAL "
                          "split -- always used for eval regardless of --on-the-fly/--no-on-the-fly "
                          "in train_reconstructor.py")
    ap.add_argument("--tokenizer-corpus-n-per-type", type=int, default=2000,
                     help="TEXT-ONLY samples per diagram type for train_tokenizer.py -- no "
                          "rendering, so this can be much larger than the val set above for "
                          "better vocabulary coverage")
    ap.add_argument("--fixed-train-n-per-type", type=int, default=0,
                     help="if >0, ALSO renders the old-style full fixed reconstructor train+val "
                          "image set to <out>/reconstructor_fixed (needed only for "
                          "`train_reconstructor.py --no-on-the-fly` comparison runs). 0 (off) by "
                          "default since the default --on-the-fly training path doesn't use it.")
    ap.add_argument("--skip-router", action="store_true", help="skip the router (classifier) dataset")
    ap.add_argument("--skip-reconstructor-val", action="store_true", help="skip the reconstructor val set")
    ap.add_argument("--skip-tokenizer-corpus", action="store_true", help="skip the tokenizer text corpus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke-test", action="store_true",
                     help="just render one test image and exit, to confirm mermaidx works")
    args = ap.parse_args()

    if args.smoke_test:
        raise SystemExit(0 if smoke_test() else 1)

    out = Path(args.out)

    if not args.skip_router:
        print("\n=== Router dataset (all diagram types + not_diagram) ===")
        generate_router_dataset(out / "router", args.router_n_per_class, args.seed)

    if not args.skip_reconstructor_val:
        print("\n=== Reconstructor VAL set (fixed, rendered -- train comes from on-the-fly generation) ===")
        generate_reconstructor_val_set(out / "reconstructor", args.reconstructor_val_n_per_type, args.seed)

    if not args.skip_tokenizer_corpus:
        print("\n=== Tokenizer corpus (text-only, no rendering) ===")
        generate_tokenizer_corpus(
            out / "reconstructor" / "tokenizer_corpus.jsonl", args.tokenizer_corpus_n_per_type, args.seed,
        )

    if args.fixed_train_n_per_type > 0:
        print("\n=== Reconstructor FIXED full train+val set (legacy -- for --no-on-the-fly only) ===")
        generate_reconstructor_dataset(out / "reconstructor_fixed", args.fixed_train_n_per_type, args.seed)

    print("\nDone. Next: make tokenizer && make train-router && make train-reconstructor")
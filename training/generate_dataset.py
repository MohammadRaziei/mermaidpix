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
        print(f"  reconstructor/{diagram_type}: {ok}/{n_per_type} rendered")

    with open(out_dir / "manifest.jsonl", "w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    print(f"  manifest -> {out_dir / 'manifest.jsonl'} ({len(manifest)} rows)")


def smoke_test():
    print("Running smoke test: rendering one tiny flowchart...")
    png = render_png("flowchart TD\n    A[Start] --> B[End]")
    if png is not None:
        with open("/tmp/mermaidx_smoketest.png", "wb") as f:
            f.write(png)
        print("OK -- mermaidx is working. Safe to run the full generation.")
        return True
    print("FAILED -- mermaidx did not produce an image. Check `make install` output.")
    return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="./data")
    ap.add_argument("--router-n-per-class", type=int, default=300)
    ap.add_argument("--reconstructor-n-per-type", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke-test", action="store_true",
                     help="just render one test image and exit, to confirm mermaidx works")
    args = ap.parse_args()

    if args.smoke_test:
        raise SystemExit(0 if smoke_test() else 1)

    out = Path(args.out)

    print("\n=== Router dataset (all diagram types + not_diagram) ===")
    generate_router_dataset(out / "router", args.router_n_per_class, args.seed)

    print("\n=== Reconstructor dataset (image -> mermaid source pairs) ===")
    generate_reconstructor_dataset(out / "reconstructor", args.reconstructor_n_per_type, args.seed)

    print("\nDone. Next: make tokenizer && make train-router && make train-reconstructor")

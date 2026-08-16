"""
roundtrip_test.py — an end-to-end sanity check, run automatically at the
end of `train_reconstructor.py` (and re-runnable standalone against any
saved checkpoint).

Takes a small, FIXED, deterministic set of Mermaid sources (one per
diagram type, same every run -- so results are comparable run over run),
renders each with mermaidx's default quickjs backend specifically
(deliberately NOT the mixed engines training might have used -- this is
meant to be a stable reference check, not an engine-robustness test),
feeds the PNG through the trained reconstructor, and writes the
(ground_truth, prediction) pairs to a report.

This is the actual real-world task the whole project exists for: hand it
a picture, get matching Mermaid code back. qualitative_samples in
train_reconstructor.py already samples something similar from held-out
*generated* val data; this is a small, fixed, eyeballable set instead.

Standalone usage:
    python roundtrip_test.py --model ./results/reconstructor/reconstructor_model.pt --tokenizer ./tokenizer
"""
from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import torch
from PIL import Image

from common.diagram_generators import DIAGRAM_BUILDERS, wrap_with_frontmatter


def build_fixed_testset(seed: int = 1234) -> list[dict]:
    """One deterministic example per diagram type. Same seed -> same
    *source text* every time this is called -- confirmed in conversation:
    target text is always byte-identical across calls, but the rendered
    PNG bytes themselves can differ slightly for a few diagram types
    (observed: flowchart/gitgraph/cynefin), most likely from mermaid.js's
    own internal auto-generated SVG element IDs (clipPaths etc.), not from
    anything under this project's control. That's fine for this file's
    purpose -- what has to stay fixed run-over-run is *which* diagrams get
    tested and what their correct answer is, not the exact pixels, and a
    few-byte SVG-internal-ID difference doesn't change what's visually on
    the page. Fixed theme/look (not randomized) for the same reason.
    Always rendered with mermaidx's own default quickjs backend,
    regardless of what --render-engines training used -- see module
    docstring."""
    import mermaidx  # local import -- keeps this file safe to import from
                      # a context (e.g. a spawned child process) that
                      # shouldn't eagerly pull in mermaidx at module load

    testset = []
    for diagram_type, builder in DIAGRAM_BUILDERS.items():
        rng = random.Random(seed)  # reset per type: deterministic and
                                    # independent of dict iteration order
        src = builder(rng)
        wrapped = wrap_with_frontmatter(src, theme="default", look="classic")
        try:
            diagram = mermaidx.render(wrapped)
            png_bytes = diagram.png(width=800, background="#ffffff")
        except Exception as e:
            print(f"[roundtrip_test] WARNING: could not render {diagram_type}: {e}")
            continue
        if png_bytes is None:
            print(f"[roundtrip_test] WARNING: {diagram_type} rendered to no image, skipping")
            continue
        testset.append({"diagram_type": diagram_type, "target": src, "png_bytes": png_bytes})
    return testset


@torch.no_grad()
def run_roundtrip_test(
    model, tokenizer, device, out_dir: Path, seed: int = 1234, max_new_tokens: int = 500,
) -> list[dict]:
    """Called directly from train_reconstructor.py's main() with the
    just-trained model still in memory (no checkpoint reload needed), or
    from this file's own main() for standalone re-runs against a saved
    checkpoint. Writes out_dir/roundtrip_report.json (machine-readable),
    out_dir/roundtrip_report.txt (human-eyeballable), and out_dir/images/
    (the rendered PNGs, so you can look at exactly what the model saw)."""
    from train_reconstructor import build_transform  # local import: avoids a circular
                                                       # import at module load time, since
                                                       # train_reconstructor.py imports
                                                       # this function too (see its main())

    transform = build_transform(train=False)
    was_training = model.training
    model.eval()

    testset = build_fixed_testset(seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)

    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")
    pad_id = tokenizer.token_to_id("<pad>")

    results = []
    for sample in testset:
        img = Image.open(io.BytesIO(sample["png_bytes"]))
        x = transform(img.convert("RGB")).unsqueeze(0).to(device)
        ids = model.generate(x, bos_id=bos_id, eos_id=eos_id, max_new_tokens=max_new_tokens)[0]
        clean_ids = [t for t in ids.tolist() if t not in (bos_id, eos_id, pad_id)]
        prediction = tokenizer.decode(clean_ids)

        img_path = images_dir / f"{sample['diagram_type']}.png"
        with open(img_path, "wb") as f:
            f.write(sample["png_bytes"])

        results.append({
            "diagram_type": sample["diagram_type"],
            "image_file": f"images/{img_path.name}",
            "ground_truth": sample["target"],
            "prediction": prediction,
            "exact_match": prediction.strip() == sample["target"].strip(),
        })

    if was_training:
        model.train()  # restore caller's mode -- this shouldn't have side effects
                        # on a training run that happens to call this mid-flow

    with open(out_dir / "roundtrip_report.json", "w") as f:
        json.dump(results, f, indent=2)

    n_match = sum(r["exact_match"] for r in results)
    lines = [f"Round-trip test (mmd -> quickjs png -> model -> mmd): "
             f"{n_match}/{len(results)} exact matches", ""]
    for r in results:
        lines.append(f"=== {r['diagram_type']}  (exact_match={r['exact_match']}) ===")
        lines.append("--- ground truth ---")
        lines.append(r["ground_truth"])
        lines.append("--- prediction ---")
        lines.append(r["prediction"])
        lines.append("")
    with open(out_dir / "roundtrip_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="./results/reconstructor/reconstructor_model.pt")
    ap.add_argument("--tokenizer", default="./tokenizer")
    ap.add_argument("--out", default="./results/reconstructor/roundtrip")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    from model import MermaidReconstructor
    from train_reconstructor import load_tokenizer

    if not Path(args.model).exists():
        raise SystemExit(f"Missing checkpoint: {args.model} -- run `make train-reconstructor` first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(Path(args.tokenizer))
    ckpt = torch.load(args.model, map_location=device)
    model = MermaidReconstructor(vocab_size=ckpt["vocab_size"], pad_id=ckpt["pad_id"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    results = run_roundtrip_test(model, tokenizer, device, Path(args.out), seed=args.seed)
    n_match = sum(r["exact_match"] for r in results)
    print(f"{n_match}/{len(results)} exact matches -- see {args.out}/roundtrip_report.txt")


if __name__ == "__main__":
    main()

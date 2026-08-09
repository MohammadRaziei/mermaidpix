"""
ocr_refine.py — EXPERIMENTAL, optional post-processing step.

Separate from the vocabulary-fix experiment on purpose (see IDEA.md,
"Fallback plan if the vocabulary fix isn't enough") -- run this on top of
whichever reconstructor checkpoint you have *in addition to* the
vocabulary-expanded retrain, not instead of it, so the two experiments
stay isolated and you can tell which one (or both) actually helped.

What it does:
    1. Takes the reconstructor's predicted Mermaid code (structure is
       usually right, per the diagnosis in IDEA.md -- labels may be wrong).
    2. Runs `pyturboocr` on the original image to get every text box it
       can actually read, independently of the reconstructor.
    3. Since the current architecture has no per-label bounding boxes to
       match against (a deliberate seq2seq design choice -- see IDEA.md),
       matches OCR'd text to predicted labels by READING ORDER (top-to-
       bottom, then left-to-right) instead of position. This is a
       heuristic, not a guarantee: it assumes the decoder emits labels in
       roughly the same order a human would read them, which held for
       every generator in common/diagram_generators.py (nodes are IDed
       and typically laid out top-down) but hasn't been verified against
       real, arbitrarily-laid-out diagrams.
    4. Substitutes the OCR'd text into each label slot, in order.

Usage:
    python ocr_refine.py my_diagram.png --router-model results/router/router_model.pt \\
        --reconstructor-model results/reconstructor/reconstructor_model.pt --tokenizer ./tokenizer

Requires: pip install pyturboocr
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from PIL import Image

from infer import load_models, route, reconstruct


def ocr_labels_in_reading_order(image_path: str, tier: str = "tiny") -> list[str]:
    """Returns every text string pyturboocr finds, sorted top-to-bottom
    then left-to-right (reading order), which is what we match against
    since we don't have per-label positions on the reconstructor's side."""
    from pyturboocr import OCR

    ocr = OCR(tier=tier)
    result = ocr.recognize_image(image_path)

    items = []
    for item in result.results:
        # box is 4 corner points; use the top-left-most corner for sorting
        xs = [p[0] for p in item.box]
        ys = [p[1] for p in item.box]
        items.append((min(ys), min(xs), item.text, item.confidence))

    items.sort(key=lambda t: (round(t[0] / 20), t[1]))  # bucket rows ~20px tall, then sort by x within a row
    return [text for _, _, text, _ in items]


def substitute_labels(mermaid_code: str, ocr_texts: list[str]) -> tuple[str, dict]:
    """Replaces each quoted label in the Mermaid source with the next
    OCR'd string, in the order labels appear in the source text. Returns
    the modified code plus a small report of what got replaced, so you can
    see exactly what changed (and judge whether the substitutions look
    like improvements or not) rather than trusting it blindly."""
    label_pattern = re.compile(r'"([^"]*)"')
    original_labels = label_pattern.findall(mermaid_code)

    report = {
        "original_labels": original_labels,
        "ocr_texts_available": ocr_texts,
        "substitutions": [],
    }

    ocr_iter = iter(ocr_texts)

    def _replace(match):
        original = match.group(1)
        try:
            new_text = next(ocr_iter)
        except StopIteration:
            report["substitutions"].append({"original": original, "replacement": None, "reason": "ran out of OCR text"})
            return match.group(0)
        report["substitutions"].append({"original": original, "replacement": new_text})
        return f'"{new_text}"'

    refined_code = label_pattern.sub(_replace, mermaid_code)
    return refined_code, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=str)
    ap.add_argument("--router-model", default="./results/router/router_model.pt")
    ap.add_argument("--reconstructor-model", default="./results/reconstructor/reconstructor_model.pt")
    ap.add_argument("--tokenizer", default="./tokenizer")
    ap.add_argument("--ocr-tier", default="tiny", choices=["tiny", "small", "medium"])
    args = ap.parse_args()

    for p in (args.router_model, args.reconstructor_model):
        if not Path(p).exists():
            raise SystemExit(f"Missing checkpoint: {p} -- run the training steps first.")
    if not Path(args.tokenizer).exists():
        raise SystemExit(f"Missing tokenizer dir: {args.tokenizer} -- run `make tokenizer` first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router, classes, reconstructor, tokenizer = load_models(
        args.router_model, args.reconstructor_model, args.tokenizer, device
    )

    image = Image.open(args.image)
    diagram_type, confidence = route(image, router, classes, device)
    print(f"Diagram type: {diagram_type}  (confidence: {confidence:.2f})")
    if diagram_type == "not_diagram":
        print("(no mermaid code -- this doesn't look like a mermaid diagram)")
        return

    raw_code = reconstruct(image, reconstructor, tokenizer, device)
    print("\n=== Reconstructor output (structure) ===")
    print(raw_code)

    try:
        ocr_texts = ocr_labels_in_reading_order(args.image, tier=args.ocr_tier)
    except ImportError:
        raise SystemExit("pyturboocr not installed -- run: pip install pyturboocr")

    print(f"\n=== OCR found {len(ocr_texts)} text regions (reading order) ===")
    for t in ocr_texts:
        print(f"  {t!r}")

    refined_code, report = substitute_labels(raw_code, ocr_texts)

    print("\n=== Refined output (labels substituted from OCR) ===")
    print(refined_code)

    print("\n=== What changed ===")
    for sub in report["substitutions"]:
        if sub.get("replacement") is None:
            print(f'  {sub["original"]!r} -> KEPT (no OCR text left to substitute)')
        elif sub["replacement"] == sub["original"]:
            print(f'  {sub["original"]!r} -> unchanged (OCR agreed)')
        else:
            print(f'  {sub["original"]!r} -> {sub["replacement"]!r}')

    if len(report["original_labels"]) != len(ocr_texts):
        print(
            f"\nNOTE: reconstructor predicted {len(report['original_labels'])} labels but "
            f"OCR found {len(ocr_texts)} text regions -- counts don't match, so the "
            "reading-order alignment above is likely off for at least some labels "
            "(e.g. edge labels like 'Yes'/'No' also get picked up by OCR and will "
            "shift every subsequent match). Treat this run's output as a rough signal, "
            "not a reliable correction, until the matching strategy is improved."
        )


if __name__ == "__main__":
    main()

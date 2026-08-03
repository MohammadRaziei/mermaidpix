"""
infer.py — Step 5. The "give me a picture, get me an answer" script.

    python infer.py my_diagram.png

Output:
    Diagram type: flowchart  (confidence: 0.94)
    Mermaid code:
    flowchart TD
        ...

Both models are loaded once at startup and kept in memory -- there's no
per-diagram-type model swapping, which was the loading-cost concern this
architecture was specifically chosen to avoid.

Requires results/router/router_model.pt and
results/reconstructor/reconstructor_model.pt (produced by `make
train-router` / `make train-reconstructor`), plus ./tokenizer/ (produced by
`make tokenizer`).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from tokenizers import ByteLevelBPETokenizer
from torchvision import transforms

from common.diagram_generators import ROUTER_CLASSES
from model import MermaidReconstructor
from train_reconstructor import IMG_TRANSFORM, load_tokenizer
from train_router import build_model as build_router_model

_ROUTER_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_models(router_path, reconstructor_path, tokenizer_dir, device):
    router_ckpt = torch.load(router_path, map_location=device)
    router = build_router_model(len(router_ckpt["classes"]))
    router.load_state_dict(router_ckpt["model_state"])
    router.to(device).eval()

    tokenizer = load_tokenizer(Path(tokenizer_dir))

    recon_ckpt = torch.load(reconstructor_path, map_location=device)
    reconstructor = MermaidReconstructor(vocab_size=recon_ckpt["vocab_size"], pad_id=recon_ckpt["pad_id"])
    reconstructor.load_state_dict(recon_ckpt["model_state"])
    reconstructor.to(device).eval()

    return router, router_ckpt["classes"], reconstructor, tokenizer


@torch.no_grad()
def route(image: Image.Image, router, classes, device):
    x = _ROUTER_TF(image.convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(router(x), dim=1)[0]
    idx = int(probs.argmax())
    return classes[idx], float(probs[idx])


@torch.no_grad()
def reconstruct(image: Image.Image, reconstructor, tokenizer, device, max_new_tokens=500):
    x = IMG_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")
    pad_id = tokenizer.token_to_id("<pad>")
    ids = reconstructor.generate(x, bos_id=bos_id, eos_id=eos_id, max_new_tokens=max_new_tokens)[0]
    clean_ids = [t for t in ids.tolist() if t not in (bos_id, eos_id, pad_id)]
    return tokenizer.decode(clean_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=str)
    ap.add_argument("--router-model", default="./results/router/router_model.pt")
    ap.add_argument("--reconstructor-model", default="./results/reconstructor/reconstructor_model.pt")
    ap.add_argument("--tokenizer", default="./tokenizer")
    args = ap.parse_args()

    for p in (args.router_model, args.reconstructor_model):
        if not Path(p).exists():
            raise SystemExit(f"Missing checkpoint: {p} -- run the training steps first (see README).")
    if not Path(args.tokenizer, ).exists():
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

    mermaid_code = reconstruct(image, reconstructor, tokenizer, device)
    print("Mermaid code:")
    print(mermaid_code)


if __name__ == "__main__":
    main()

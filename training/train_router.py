"""
train_router.py — Step 3 (run via `make train-router`).

Diagram-type classifier: not_diagram (class 0) vs. the 8 supported Mermaid
diagram types. Architecture: MobileNetV3-Small, ImageNet-pretrained,
fine-tuned -- ~2.5M params, built for exactly this kind of fast
classification workload (mobile/edge deployment), so it's the natural
choice when the stated priority is speed over squeezing out the last
fraction of a percent of accuracy a bigger backbone (ResNet50 etc.) might
give on a dataset this size.

Writes per-epoch metrics, the final confusion matrix, and the model
checkpoint into results/router/ (picked up by `make package` at the end).

Usage:
    python train_router.py --data ./data/router --epochs 15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from common.diagram_generators import ROUTER_CLASSES


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def get_loaders(data_dir: Path, batch_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=val_tf)

    assert train_ds.classes == sorted(ROUTER_CLASSES), (
        f"class order mismatch: {train_ds.classes} vs {sorted(ROUTER_CLASSES)} -- "
        "ImageFolder sorts alphabetically; if this fails, a class folder name "
        "changed somewhere without updating ROUTER_CLASSES."
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, train_ds.classes


def evaluate(model, loader, device, n_classes):
    model.eval()
    correct, total = 0, 0
    confusion = torch.zeros(n_classes, n_classes, dtype=torch.int64)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            for t, p in zip(y.cpu(), pred.cpu()):
                confusion[t, p] += 1
    return correct / total, confusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="./data/router")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--results-dir", type=str, default="./results/router")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_loader, val_loader, classes = get_loaders(Path(args.data), args.batch_size)
    print("classes (alphabetical, this is the label index order):", classes)

    model = build_model(len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_acc, confusion = evaluate(model, val_loader, device, len(classes))
        print(f"epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state": model.state_dict(), "classes": classes},
                       results_dir / "router_model.pt")

    with open(results_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    confusion_list = confusion.tolist()
    summary = {
        "classes": classes,
        "best_val_acc": best_acc,
        "final_confusion_matrix": confusion_list,
        "num_epochs": args.epochs,
        "num_train_images": len(train_loader.dataset),
        "num_val_images": len(val_loader.dataset),
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest val accuracy: {best_acc:.4f}")
    print("Confusion matrix (rows=true, cols=predicted):")
    print("classes:", classes)
    print(confusion)
    print(f"\nSaved model + history + summary to {results_dir}/")


if __name__ == "__main__":
    main()

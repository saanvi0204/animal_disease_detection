"""
train.py
--------
Trains EfficientNet-B0 on the animal disease dataset.
Two-phase training: freeze backbone → train head → unfreeze all → full fine-tune.

"""

import os
import sys
import argparse
import json
import time

import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import DataLoader
import timm

sys.path.insert(0, os.path.dirname(__file__))
from dataset import AnimalDiseaseDataset
from utils   import get_transforms, get_class_weights, EarlyStopping, save_training_curves


def parse_args():
    p = argparse.ArgumentParser(description="Train EfficientNet-B0 for animal disease detection")
    p.add_argument("--species",     required=True, choices=["cats", "dogs"])
    p.add_argument("--data_dir",    type=str, default=None,     help="Overrides default data/species/")
    p.add_argument("--model_dir",   type=str, default="models/")
    p.add_argument("--results_dir", type=str, default=None,     help="Overrides default results/species/")
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--head_epochs", type=int, default=5,        help="Epochs to train head-only before unfreezing")
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--head_lr",     type=float, default=1e-3,   help="LR for head-only phase")
    p.add_argument("--img_size",    type=int, default=224)
    p.add_argument("--patience",    type=int, default=7)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def run_epoch(model, loader, criterion, optimizer, scaler, device, training: bool):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad()
                with amp.autocast(device_type=device.type):
                    logits = model(images)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with amp.autocast(device_type=device.type):
                    logits = model(images)
                    loss   = criterion(logits, labels)

            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    return total_loss / total, 100.0 * correct / total


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir    = args.data_dir    or f"data/{args.species}/"
    results_dir = args.results_dir or f"results/{args.species}/"

    print(f"\n{'='*55}")
    print(f"  Species : {args.species}")
    print(f"  Data    : {data_dir}")
    print(f"  Device  : {device}")
    print(f"{'='*55}\n")

    # ── Datasets ──────────────────────────────────────────────────
    train_ds = AnimalDiseaseDataset(data_dir, "train", get_transforms("train", args.img_size))
    val_ds   = AnimalDiseaseDataset(data_dir, "val",   get_transforms("val",   args.img_size))

    num_classes = len(train_ds.classes)
    print(f"\nClasses ({num_classes}): {train_ds.classes}\n")

    # Save class mapping — consumed by evaluate, gradcam, inference, and app
    os.makedirs(args.model_dir, exist_ok=True)
    mapping_path = os.path.join(args.model_dir, f"class_mapping_{args.species}.json")
    with open(mapping_path, "w") as f:
        json.dump({"classes": train_ds.classes, "class_to_idx": train_ds.class_to_idx}, f, indent=2)
    print(f"Saved class mapping → {mapping_path}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
    model = model.to(device)

    weights   = get_class_weights(train_ds, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    scaler    = torch.amp.GradScaler()

    checkpoint_name = f"best_model_{args.species}.pth"
    stopper = EarlyStopping(
        patience=args.patience,
        path=os.path.join(args.model_dir, checkpoint_name),
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # ══════════════════════════════════════════════════════════════
    # PHASE 1 — head only
    # ══════════════════════════════════════════════════════════════
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"\nPhase 1 — trainable: {trainable:,} / {total_p:,} params (head only)")
    print(f"\n  {'Epoch':<8} {'Tr Loss':<10} {'Tr Acc':<10} {'Va Loss':<10} {'Va Acc':<10}")
    print(f"  {'-'*50}")

    optimizer_head = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.head_lr, weight_decay=1e-4,
    )

    for epoch in range(1, args.head_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer_head, scaler, device, True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, optimizer_head, scaler, device, False)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        print(f"  {epoch:02d}/{args.head_epochs:<6} {tr_loss:<10.4f} {tr_acc:<10.1f} {va_loss:<10.4f} {va_acc:<10.1f} {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2 — full fine-tune
    # ══════════════════════════════════════════════════════════════
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nPhase 2 — trainable: {trainable:,} params (all layers unfrozen)")
    print(f"\n  {'Epoch':<8} {'Tr Loss':<10} {'Tr Acc':<10} {'Va Loss':<10} {'Va Acc':<10}")
    print(f"  {'-'*50}")

    optimizer_full = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler      = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_full, mode="min", factor=0.5, patience=3
    )

    for epoch in range(args.head_epochs + 1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer_full, scaler, device, True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, optimizer_full, scaler, device, False)
        scheduler.step(va_loss)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        print(f"  {epoch:02d}/{args.epochs:<6} {tr_loss:<10.4f} {tr_acc:<10.1f} {va_loss:<10.4f} {va_acc:<10.1f} {time.time()-t0:.1f}s")
        stopper(va_loss, model)
        if stopper.early_stop:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    # ── Post-training ─────────────────────────────────────────────
    os.makedirs(results_dir, exist_ok=True)
    save_training_curves(
        history["train_loss"], history["val_loss"],
        history["train_acc"],  history["val_acc"],
        save_dir=results_dir, head_epochs=args.head_epochs,
    )
    with open(os.path.join(results_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Best val loss : {stopper.best_loss:.4f}")
    print(f"  Best val acc  : {max(history['val_acc']):.1f}%")
    print(f"  Checkpoint    : {stopper.path}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    train(parse_args())

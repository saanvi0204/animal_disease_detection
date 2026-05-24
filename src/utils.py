"""
utils.py
--------
Shared helpers: transforms, class-weight computation, and plot saving.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from sklearn.utils.class_weight import compute_class_weight


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(split: str = "train", img_size: int = 224):
    """
    Augmentation on train split only.
    Val and test use a clean deterministic resize pipeline.
    """
    if split == "train":
        return T.Compose([
            T.Resize((img_size + 32, img_size + 32)),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def denormalize(tensor):
    """Reverse ImageNet normalization for visualization."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)


def get_class_weights(dataset, device: torch.device) -> torch.Tensor:
    """
    Compute balanced class weights to counter class imbalance.
    Passed directly to CrossEntropyLoss(weight=...).
    """
    labels = np.array(dataset.get_labels())
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    print("Class weights:")
    for cls, w in zip(dataset.classes, weights):
        bar = "█" * int(w * 20)
        print(f"  {cls:<35} {w:.4f}  {bar}")
    return torch.tensor(weights, dtype=torch.float).to(device)


class EarlyStopping:
    """
    Stops training when val loss stops improving.
    Saves the best checkpoint automatically.

    Args:
        patience  : epochs to wait before stopping.
        min_delta : minimum improvement to reset counter.
        path      : where to save the best model checkpoint.
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4, path: str = "models/best_model.pth"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.path       = path
        self.counter    = 0
        self.best_loss  = None
        self.early_stop = False

    def __call__(self, val_loss: float, model) -> bool:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            torch.save(model.state_dict(), self.path)
            print(f"  ✔ Checkpoint saved (val_loss={val_loss:.4f})")
        else:
            self.counter += 1
            print(f"  No improvement {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def save_training_curves(train_losses, val_losses, train_accs, val_accs,
                         save_dir: str = "results", head_epochs: int = 5):
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(train_losses, label="Train", linewidth=2)
    ax1.plot(val_losses,   label="Val",   linewidth=2, linestyle="--")
    ax1.axvline(x=head_epochs, color="gray", linestyle=":", linewidth=1.5, label="Unfreeze")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("CrossEntropy")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(train_accs, label="Train", linewidth=2)
    ax2.plot(val_accs,   label="Val",   linewidth=2, linestyle="--")
    ax2.axvline(x=head_epochs, color="gray", linestyle=":", linewidth=1.5, label="Unfreeze")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves → {out}")

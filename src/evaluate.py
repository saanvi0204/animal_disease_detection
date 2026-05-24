"""
evaluate.py
-----------
Full test-set evaluation: classification report, confusion matrix, ROC curves.
Normalises softmax probabilities before ROC to guard against numerical drift.
"""

import os
import sys
import argparse
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import timm

from torch import amp
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from dataset import AnimalDiseaseDataset
from utils   import get_transforms


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained model on test set")
    p.add_argument("--species",     required=True, choices=["cats", "dogs"])
    p.add_argument("--data_dir",    type=str, default=None)
    p.add_argument("--model_dir",   type=str, default="models/")
    p.add_argument("--results_dir", type=str, default=None)
    p.add_argument("--checkpoint",  type=str, default=None,  help="Defaults to best_model_{species}.pth")
    p.add_argument("--img_size",    type=int, default=224)
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def load_model(checkpoint_path, num_classes, device):
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.to(device).eval()


def get_predictions(model, loader, device):
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            with amp.autocast(device_type=device.type):
                logits = model(images)
            probs = torch.softmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    labels = np.array(all_labels)
    preds  = np.array(all_preds)
    probs  = np.asarray(all_probs, dtype=np.float64)
    probs  = probs / (probs.sum(axis=1, keepdims=True) + 1e-12)
    return labels, preds, probs


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted", ylabel="True", title="Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix → {save_path}")


def plot_roc_curves(labels, probs, class_names, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(class_names):
        binary      = (labels == i).astype(int)
        fpr, tpr, _ = roc_curve(binary, probs[:, i])
        auc         = roc_auc_score(binary, probs[:, i])
        ax.plot(fpr, tpr, lw=2, label=f"{cls}  (AUC = {auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curves (One-vs-Rest)", xlim=[0,1], ylim=[0,1.02])
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curve → {save_path}")


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir    = args.data_dir    or f"data/{args.species}/"
    results_dir = args.results_dir or f"results/{args.species}/"
    checkpoint  = args.checkpoint  or f"best_model_{args.species}.pth"
    os.makedirs(results_dir, exist_ok=True)

    mapping_path = os.path.join(args.model_dir, f"class_mapping_{args.species}.json")
    with open(mapping_path) as f:
        mapping     = json.load(f)
    class_names = mapping["classes"]
    num_classes  = len(class_names)

    print(f"\nEvaluating [{args.species}] on TEST set | Classes: {class_names}\n")

    test_ds = AnimalDiseaseDataset(data_dir, "test", get_transforms("test", args.img_size))
    loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    model  = load_model(os.path.join(args.model_dir, checkpoint), num_classes, device)
    labels, preds, probs = get_predictions(model, loader, device)

    report = classification_report(labels, preds, target_names=class_names, digits=4)
    print("── Classification Report ──────────────────────────────")
    print(report)
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    try:
        macro_auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        print(f"Macro ROC-AUC (OvR): {macro_auc:.4f}")
    except Exception as e:
        print(f"ROC-AUC skipped: {e}")

    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, class_names, os.path.join(results_dir, "confusion_matrix.png"))
    plot_roc_curves(labels, probs, class_names, os.path.join(results_dir, "roc_curve.png"))
    print("\nEvaluation complete.\n")


if __name__ == "__main__":
    evaluate(parse_args())

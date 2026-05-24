"""
gradcam.py
----------
Generates Grad-CAM heatmap overlays for a batch of test images
using the pytorch-grad-cam library.
"""

import os
import sys
import argparse
import json
import random

import numpy as np
import torch
import timm
import matplotlib.pyplot as plt

from torch import amp
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.insert(0, os.path.dirname(__file__))
from dataset import AnimalDiseaseDataset
from utils   import get_transforms, denormalize

CAT_CLASSES = [
    "cat with abscesses",
    "cat with feline acne",
    "cat with mange",
    "healthy cat"
]
DOG_CLASSES = [
    "dog with alopecia",
    "dog with bacterial dermatosis",
    "dog with lupus",
    "healthy dog"
]

def parse_args():
    p = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps")
    p.add_argument("--species",     required=True, choices=["cats", "dogs"])
    p.add_argument("--data_dir",    type=str, default=None)
    p.add_argument("--model_dir",   type=str, default="models/")
    p.add_argument("--results_dir", type=str, default=None)
    p.add_argument("--checkpoint",  type=str, default=None)
    p.add_argument("--split",       type=str, default="test", choices=["train","val","test"])
    p.add_argument("--num_images",  type=int, default=16)
    p.add_argument("--img_size",    type=int, default=224)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def load_model(checkpoint_path, num_classes, device):
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.to(device).eval()


def get_target_layer(model):
    """
    For EfficientNet-B0 (timm), the last convolutional block is
    model.blocks[-1][-1].conv_pwl — a good Grad-CAM target.
    """
    return [model.blocks[-1][-1].conv_pwl]


def generate_gradcam(model, cam, image_tensor, class_names, device):
    input_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1)
    pred_idx   = probs.argmax().item()
    confidence = probs[0, pred_idx].item()
    all_probs  = {class_names[i]: probs[0, i].item() for i in range(len(class_names))}

    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    rgb_img   = denormalize(image_tensor).permute(1, 2, 0).numpy().astype(np.float32)
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return class_names[pred_idx], confidence, all_probs, cam_image, rgb_img


def save_single(rgb_img, cam_image, true_label, pred_label, confidence, save_path):
    """Saves original | Grad-CAM overlay side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(rgb_img)
    axes[0].set_title(f"Original\nTrue: {true_label}", fontsize=10)
    axes[0].axis("off")

    correct = true_label == pred_label
    color   = "green" if correct else "red"
    axes[1].imshow(cam_image)
    axes[1].set_title(
        f"Grad-CAM\nPred: {pred_label}  ({confidence:.0%})",
        fontsize=10, color=color,
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()


def save_grid(results, class_names, save_path, cols=4):
    """Save a summary grid of all Grad-CAM results."""
    n    = len(results)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 5, rows * 3))
    axes = np.array(axes).reshape(rows, cols * 2)

    for idx, (rgb, cam_img, true_cls, pred_cls, conf) in enumerate(results):
        r, c    = divmod(idx, cols)
        correct = true_cls == pred_cls
        axes[r, c*2].imshow(rgb);     
        axes[r, c*2].axis("off")
        axes[r, c*2].set_title(f"True: {true_cls}", fontsize=8)
        axes[r, c*2+1].imshow(cam_img); 
        axes[r, c*2+1].axis("off")
        axes[r, c*2+1].set_title(f"Pred: {pred_cls} ({conf:.0%})", fontsize=8,
                                  color="green" if correct else "red")

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c*2].axis("off")
        axes[r, c*2+1].axis("off")

    title = "Grad-CAM"

    if class_names:
        first = str(class_names[0]).lower()
        if "cat" in first:
            title += " — Cats"
        elif "dog" in first:
            title += " — Dogs"
    plt.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()


def run(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir    = args.data_dir    or f"data/{args.species}/"
    results_dir = args.results_dir or f"results/{args.species}/gradcam_outputs/"
    checkpoint  = args.checkpoint  or f"best_model_{args.species}.pth"
    os.makedirs(results_dir, exist_ok=True)

    mapping_path = os.path.join(args.model_dir, f"class_mapping_{args.species}.json")
    with open(mapping_path) as f:
        class_names = json.load(f)["classes"]
    num_classes = len(class_names)

    dataset = AnimalDiseaseDataset(args.data_dir, args.split, get_transforms("test", args.img_size))

    dataset = AnimalDiseaseDataset(data_dir, args.split, get_transforms("test", args.img_size))
    model   = load_model(os.path.join(args.model_dir, checkpoint), num_classes, device)
    cam     = GradCAM(model=model, target_layers=[model.blocks[-1][-1].conv_pwl])

    n       = min(args.num_images, len(dataset))
    indices = random.sample(range(len(dataset)), n)

    print(f"\nGrad-CAM [{args.species}] — {n} images from [{args.split}] split\n")

    results          = []
    samples_by_class = {cls: [] for cls in class_names}

    for i, idx in enumerate(indices):
        image_tensor, true_label = dataset[idx]
        true_cls = class_names[true_label]
        pred_cls, conf, all_p, cam_img, rgb_img = generate_gradcam(
            model, cam, image_tensor, class_names, device
        )
        correct = "✓" if true_cls == pred_cls else "✗"
        print(f"  [{i+1:02d}/{n}] True: {true_cls:<30} Pred: {pred_cls:<30} ({conf:.1%}) {correct}")

        save_path = os.path.join(results_dir, f"gradcam_{i+1:02d}_{true_cls.replace(' ','_')}.png")
        save_single(rgb_img, cam_img, true_cls, pred_cls, conf, save_path)

        results.append((rgb_img, cam_img, true_cls, pred_cls, conf, all_p))
        if len(samples_by_class[true_cls]) < 2:
            samples_by_class[true_cls].append((image_tensor, true_cls))

    grid_path = os.path.join(results_dir, "gradcam_summary_grid.png")
    save_grid(results, CAT_CLASSES, save_path=os.path.join(args.results_dir, "cats_gradcam.png"))
    save_grid(results, DOG_CLASSES, save_path=os.path.join(args.results_dir, "dogs_gradcam.png"))

    print(f"\nSaved summary grid → {grid_path}")
    print("Grad-CAM generation complete.\n")


if __name__ == "__main__":
    run(parse_args())

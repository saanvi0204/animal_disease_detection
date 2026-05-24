"""
inference.py
------------
Single-image inference with Grad-CAM overlay.
Used internally by the Streamlit app and for quick CLI testing.
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import timm
import matplotlib.pyplot as plt

from torch import amp
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.insert(0, os.path.dirname(__file__))
from utils import get_transforms, denormalize


def load_model_and_classes(model_dir: str, species: str, device: torch.device):
    mapping_path = os.path.join(model_dir, f"class_mapping_{species}.json")
    with open(mapping_path) as f:
        mapping     = json.load(f)
    class_names = mapping["classes"]
    num_classes  = len(class_names)

    checkpoint = os.path.join(model_dir, f"best_model_{species}.pth")
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device).eval(), class_names


def predict(image: Image.Image, model, class_names: list, device: torch.device, img_size: int = 224):
    """
    Run inference on a PIL image.

    Returns:
        pred_class (str)   : predicted class name
        confidence (float) : confidence in [0,1]
        all_probs  (dict)  : {class_name: prob}
        cam_image  (ndarray H×W×3 uint8): Grad-CAM overlay
        rgb_img    (ndarray H×W×3 float32 [0,1]): original image
    """
    transform    = get_transforms("test", img_size)
    image_tensor = transform(image.convert("RGBA").convert("RGB"))
    input_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        with amp.autocast(device_type=device.type):
            logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx   = probs.argmax().item()
    pred_class = class_names[pred_idx]
    confidence = probs[pred_idx].item()
    all_probs  = {cls: probs[i].item() for i, cls in enumerate(class_names)}

    target_layer = [model.blocks[-1][-1].conv_pwl]
    cam          = GradCAM(model=model, target_layers=target_layer)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

    rgb_img   = denormalize(image_tensor).permute(1, 2, 0).numpy().astype(np.float32)
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return pred_class, confidence, all_probs, cam_image, rgb_img


def parse_args():
    p = argparse.ArgumentParser(description="Single-image inference with Grad-CAM")
    p.add_argument("--species",    required=True, choices=["cats", "dogs"])
    p.add_argument("--image",      type=str, required=True)
    p.add_argument("--model_dir",  type=str, default="models/")
    p.add_argument("--img_size",   type=int, default=224)
    p.add_argument("--save",       type=str, default=None)
    return p.parse_args()



def run_cli(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model_and_classes(args.model_dir, args.species, device)
    image = Image.open(args.image)
    pred_class, confidence, all_probs, cam_image, rgb_img = predict(
        image, model, class_names, device, args.img_size
    )

    print(f"\n{'─'*45}")
    print(f"  Species    : {args.species}")
    print(f"  Image      : {args.image}")
    print(f"  Prediction : {pred_class}")
    print(f"  Confidence : {confidence:.1%}")
    print(f"  All probs  :")
    for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"    {cls:<35} {bar:<30} {prob:.1%}")
    print(f"{'─'*45}\n")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(rgb_img); axes[0].set_title("Original", fontsize=11); axes[0].axis("off")
    axes[1].imshow(cam_image)
    axes[1].set_title(f"Grad-CAM  →  {pred_class} ({confidence:.0%})", fontsize=11)
    axes[1].axis("off")
    plt.tight_layout()

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    run_cli(parse_args())

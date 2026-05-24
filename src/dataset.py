"""
dataset.py
----------
Custom PyTorch Dataset for animal/bird disease classification.
"""

import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class AnimalDiseaseDataset(Dataset):
    """
    Loads images from a split/class directory structure.

    Args:
        root_dir (str | Path): Path to species root (e.g. data/cats or data/dogs).
        split    (str)        : One of 'train', 'val', 'test'.
        transform             : torchvision transform pipeline.
    """

    def __init__(self, root_dir: str, split: str = "train", transform=None):
        self.split_dir = Path(root_dir) / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.transform = transform

        self.classes = sorted(
            d.name for d in self.split_dir.iterdir() if d.is_dir()
        )
        if not self.classes:
            raise ValueError(f"No class subdirectories found in {self.split_dir}")

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        self.samples = self._load_samples()
        print(
            f"[{split.upper()}] {len(self.samples)} images | "
            f"{len(self.classes)} classes: {self.classes}"
        )


    def _load_samples(self):
        samples = []
        for cls in self.classes:
            cls_dir = self.split_dir / cls
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in VALID_EXTENSIONS:
                    samples.append((img_path, self.class_to_idx[cls]))
        return samples


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGBA").convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


    def get_labels(self):
        """Return flat list of integer labels (used for class-weight computation)."""
        return [label for _, label in self.samples]


    def class_counts(self):
        """Return dict of {class_name: count}."""
        from collections import Counter
        counts = Counter(label for _, label in self.samples)
        return {self.idx_to_class[k]: v for k, v in sorted(counts.items())}

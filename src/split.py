"""
split.py
--------
One-time dataset splitting utility. Run ONCE before training.
Split is frozen after this — do not re-run.

Usage:
    # Cats (balanced — simple ratio split)
    python src/split.py --species cats --src raw_cats/ --out data/cats/

    # Dogs (imbalanced — bacterial_dermatosis capped)
    python src/split.py --species dogs --src raw_dogs/ --out data/dogs/
"""

import argparse
import random
import shutil
from pathlib import Path

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Per-class caps for dogs: (max_train, max_val, max_test)
# None = use default ratio
DOG_CAPS = {
    "dog with bacterial dermatosis": (70, 18, 18),
    "dog with lupus":                (None, None, None),
    "dog with alopecia":             (None, None, None),
    "healthy dog":                   (None, None, None),
}

DEFAULT_RATIOS = (0.71, 0.16, 0.13)


def split_cats(src_dir: Path, out_dir: Path, seed: int = 42):
    """Simple ratio split — cats dataset is roughly balanced."""
    random.seed(seed)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    print(f"  {'Class':<20} {'Train':>6} {'Val':>6} {'Test':>6}")
    print(f"  {'-'*42}")

    for cls_dir in sorted(src_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        images = sorted([f for f in cls_dir.iterdir() if f.suffix.lower() in VALID_EXT])
        random.shuffle(images)
        n       = len(images)
        n_train = int(n * DEFAULT_RATIOS[0])
        n_val   = int(n * DEFAULT_RATIOS[1])
        n_test  = n - n_train - n_val

        for split_name, imgs in zip(
            ["train", "val", "test"],
            [images[:n_train], images[n_train:n_train + n_val], images[n_train + n_val:]]
        ):
            dest = out_dir / split_name / cls_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, dest / img.name)

        print(f"  {cls_dir.name:<20} {n_train:>6} {n_val:>6} {n_test:>6}")


def split_dogs(src_dir: Path, out_dir: Path, seed: int = 42):
    """
    Per-class split with caps to prevent bacterial_dermatosis from
    dominating (~150 images vs ~70-90 for other classes).
    """
    random.seed(seed)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    print(f"  {'Class':<35} {'Train':>6} {'Val':>6} {'Test':>6}  {'Note'}")
    print(f"  {'-'*65}")

    for cls_dir in sorted(src_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name.strip()
        images   = sorted([f for f in cls_dir.iterdir() if f.suffix.lower() in VALID_EXT])
        random.shuffle(images)
        n = len(images)

        cap = DOG_CAPS.get(cls_name, (None, None, None))
        if None not in cap:
            max_train, max_val, max_test = cap
            images  = images[:max_train + max_val + max_test]
            n_train = min(max_train, len(images))
            n_val   = min(max_val,   len(images) - n_train)
            n_test  = len(images) - n_train - n_val
            note    = "← capped"
        else:
            n_train = int(n * DEFAULT_RATIOS[0])
            n_val   = int(n * DEFAULT_RATIOS[1])
            n_test  = n - n_train - n_val
            note    = ""

        for split_name, imgs in zip(
            ["train", "val", "test"],
            [images[:n_train],
             images[n_train:n_train + n_val],
             images[n_train + n_val:n_train + n_val + n_test]]
        ):
            dest = out_dir / split_name / cls_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, dest / img.name)

        print(f"  {cls_name:<35} {n_train:>6} {n_val:>6} {n_test:>6}  {note}")


def main():
    p = argparse.ArgumentParser(description="Split raw images into train/val/test")
    p.add_argument("--species", required=True, choices=["cats", "dogs"])
    p.add_argument("--src",     required=True, help="Raw images directory")
    p.add_argument("--out",     required=True, help="Output split directory")
    p.add_argument("--seed",    type=int, default=42)
    args = p.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    print(f"\nSplitting {args.species} dataset: {src} → {out}\n")

    if args.species == "cats":
        split_cats(src, out, args.seed)
    else:
        split_dogs(src, out, args.seed)

    print(f"\nDone.\n")


if __name__ == "__main__":
    main()

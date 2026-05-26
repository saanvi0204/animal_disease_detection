# 🐾 VetVision: Explainable Animal Disease Detection

> Deep learning system for detecting skin diseases in cats and dogs using **EfficientNet-B0** with **Grad-CAM** visual explanations.

---

## Problem

Skin and surface diseases are among the most common health issues in domestic animals. Early visual detection is critical for timely treatment, yet it typically requires trained veterinarians. This project explores whether a deep learning classifier can automate that detection — and, crucially, *explain* its decision by highlighting the affected region.

---

## Method

Traditional object detection (Faster R-CNN, YOLO) requires thousands of manually annotated bounding boxes. This project instead uses:

1. **Image Classification** — EfficientNet-B0 fine-tuned via transfer learning
2. **Grad-CAM** — Gradient-weighted Class Activation Mapping that produces a heatmap showing *which pixels drove the prediction*, without any annotation

```
Input Image
     ↓
EfficientNet-B0 (freeze → unfreeze training)
     ↓
Disease Prediction  +  Confidence Scores
     ↓
Grad-CAM Heatmap
     ↓
Overlay on Original Image
```

---

## Architecture

| Component | Choice | Why |
|-----------|--------|-----|
| Backbone | EfficientNet-B0 | Strong accuracy/size tradeoff; converges on small datasets |
| Pretrained weights | ImageNet | Texture/pattern features transfer well to skin conditions |
| Training strategy | Freeze head → unfreeze all | Prevents pretrained weights being corrupted by noisy early gradients |
| Loss | CrossEntropyLoss (weighted) | Handles class imbalance (especially dogs) |
| Optimizer (Phase 1) | AdamW lr=1e-3 | Head-only, higher LR safe |
| Optimizer (Phase 2) | AdamW lr=1e-4 | Full fine-tune, lower LR |
| LR scheduler | ReduceLROnPlateau | Adapts to plateau without manual tuning |
| Mixed precision | torch.amp | ~2× speedup on GPU |
| Explainability | Grad-CAM (pytorch-grad-cam) | Visualises final conv block activations |

---

## Datasets

### Cats

| Class | Raw images |
|-------|-----------|
| healthy | ~90 |
| mange | ~50 |
| deline acne | ~55 |
| abscess | ~60 |

Split: 71% / 16% / 13% (train / val / test)

### Dogs

| Class | Raw images | Note |
|-------|-----------|------|
| healthy dog | ~90 | |
| dog with bacterial dermatosis | ~150 | Capped at 70/18/18 in split |
| dog with lupus | ~70 | |
| dog with alopecia | ~90 | |

Per-class split with caps to prevent bacterial_dermatosis dominating.
Residual imbalance handled by weighted CrossEntropyLoss.

### Collection

- Images scraped using the "Download All Images" Chrome extension
- 100–300 images per class recommended

---


## Results

| Species | Test Accuracy | Macro F1 | Macro ROC-AUC |
|---------|--------------|----------|----------------|
| Cats | 0.6579 | 0.6547  | 0.8716 |
| Dogs | 0.5370 | 0.5010 | 0.7825 |

### Observations
- Performance is likely constrained by the small dataset size and class imbalance, which reduce the model’s ability to learn robust disease-specific features.
- Results may improve with a larger and more balanced dataset, stronger augmentation, and additional hyperparameter tuning.


---

## Project Structure

```
animal-disease-detection/
│
├── src/
│   ├── dataset.py      ← PyTorch Dataset 
│   ├── split.py        ← one-time dataset splitter (cats + dogs)
│   ├── train.py        ← two-phase freeze/unfreeze training loop
│   ├── evaluate.py     ← test-set metrics + confusion matrix + ROC
│   ├── gradcam.py      ← batch Grad-CAM 
│   ├── inference.py    ← single-image inference (used by app)
│   └── utils.py        ← transforms, class weights, early stopping
│
├── app/
│   └── streamlit_app.py   ← species selector + upload + heatmap
│
├── models/
│   ├── best_model_cats.pth
│   ├── best_model_dogs.pth
│   ├── class_mapping_cats.json
│   └── class_mapping_dogs.json
│
├── results/
│   ├── cats/
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   ├── training_curves.png
│   │   └── gradcam_outputs/
│   └── dogs/
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── training_curves.png
│       └── gradcam_outputs/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/saanvi0204/animal_disease_detection.git
cd animal-disease-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify pretrained models are available

The repository already includes trained checkpoints:

```plaintext
models/
├── best_model_cats.pth
├── best_model_dogs.pth
├── class_mapping_cats.json
└── class_mapping_dogs.json
```

No training is required.

### 4. Run single-image inference

Cats:

```bash
python src/inference.py \
    --species cats \
    --image path/to/cat_image.jpg
```

Dogs:

```bash
python src/inference.py \
    --species dogs \
    --image path/to/dog_image.jpg
```

Example output:

```plaintext
Prediction: cat with mange
Confidence: 92.4%
```

### 5. Launch the Streamlit application

```bash
streamlit run app/streamlit_app.py
```

Open:

```plaintext
http://localhost:8501
```

Then:

- Select species (Cats / Dogs)
- Upload an image
- View prediction
- View Grad-CAM heatmap

---

## Optional: Reproduce Training

Training scripts are included for reproducibility but are not required.

Prepare datasets and run:

```bash
python src/train.py --species cats
python src/train.py --species dogs
```

Optional evaluation:

```bash
python src/evaluate.py --species cats
python src/evaluate.py --species dogs
```

Optional Grad-CAM generation:

```bash
python src/gradcam.py --species cats
python src/gradcam.py --species dogs
```

## Future Work

- Collect more images (target 150–200 per class)
- Extend to cows and chickens as additional species-specific models

---

## Disclaimer

For research and educational purposes only. Not a substitute for professional veterinary diagnosis.

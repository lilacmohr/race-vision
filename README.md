# race-vision

Detecting and classifying running gear in race photos using YOLOv8 and CLIP.

## Project Overview

A multi-phase computer vision pipeline that detects running gear in race photos, reads bib numbers via OCR, and classifies gear attributes (brand, color). The architecture is designed to transfer directly to decorated apparel detection.

## Phases

1. **Environment Setup** — Ultralytics, CLIP, EDA libraries, Roboflow account; local GPU via Apple MPS
2. **Dataset Collection** — Roboflow Universe datasets + scraped race photos (Flickr, SmugMug); metadata tracked per image
3. **EDA** — Duplicate detection (`imagehash`), CLIP embeddings + UMAP diversity analysis, image statistics
4. **Labeling** — Roboflow web labeler; starting with `race_bib`, then expanding class by class
5. **YOLO Training** — Fine-tune YOLOv8n → YOLOv8s/m; evaluate mAP50 / mAP50-95; iterate on data before hyperparameters
6. **Bib OCR** — Crop bib detections → EasyOCR → race number string
7. **Attribute Classifier** — ResNet18/EfficientNet-B0 on cropped detections; predict brand and color per class
8. **Inference Pipeline** — Structured output with bounding boxes, attribute labels, and confidence scores
9. **Project 2: Decorated Apparel** — Apply the same pipeline to garment imagery and print zone detection

## Detection Classes

| Class | Notes |
|---|---|
| `race_bib` | First class to label — most visually distinctive |
| `shoe` | Second class |
| `top` | Singlet, shirt, jacket |
| `shorts` / `tights` | |
| `gps_watch` | |
| `sunglasses` | |
| `cap` / `visor` | |
| `hydration_vest` | |

## Getting Started

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the dataset

**Option A — use the existing labeled dataset (recommended)**

The labeled dataset is hosted on Roboflow (CC BY 4.0):

1. Go to [race-vision on Roboflow Universe](https://universe.roboflow.com/lilacs-workspace/race-vision-rmatv)
2. Click **Download Dataset** → format: **YOLOv8** → **download zip to computer**
3. Unzip into `data/labeled/` so the structure is:
   ```
   data/labeled/race-vision.v1i.yolov8/
     data.yaml
     train/images/   train/labels/
     valid/images/   valid/labels/
     test/images/    test/labels/
   ```

The folder must be named `race-vision.v1i.yolov8/` — the notebooks reference this path directly.

**Option B — build your own dataset**

1. **Collect raw images** — aim for 150–300 images with varied lighting, distances, and race types. Good sources:
   - Flickr: search `marathon finish line`, `running race bib`, `half marathon crowd` — filter for CC-licensed photos
   - [Roboflow Universe](https://universe.roboflow.com) — search for existing running/race datasets to merge
   - Your own race photos
   
   Place images in `data/raw/`. Run `01_eda.ipynb` to check for duplicates and dataset diversity before labeling.

2. **Label with Roboflow** — create a free account at [roboflow.com](https://roboflow.com), upload your images, and draw bounding boxes for `bib` (or whatever classes you want). Use Roboflow's auto-label (SAM2-backed) for a first pass, then review every label manually. Target 80/10/10 train/val/test split with images from different races in each split.

3. **Export** — export in YOLOv8 format and place the output in `data/labeled/` as above. Update `DATASET_DIR` in each notebook if you use a different folder name.

### 3. Run the notebooks in order

| Notebook | What it does | Requires |
|---|---|---|
| `01_eda.ipynb` | Duplicate detection, CLIP embeddings, UMAP diversity visualization | `data/raw/` — see note below |
| `03_training.ipynb` | YOLOv8n fine-tuning | Roboflow dataset |
| `04_evaluation.ipynb` | YOLOv8 evaluation — mAP, false positives/negatives | `03` trained weights |
| `05_rfdetr_training.ipynb` | RF-DETR training | Roboflow dataset |
| `06_rfdetr_evaluation.ipynb` | RF-DETR evaluation and model comparison | `05` trained weights |
| `07_image_analysis.ipynb` | Image attribute analysis (brightness, sharpness, contrast vs. confidence) | `03` and `05` trained weights |

Notebooks 04, 06, and 07 load weights produced by training runs — run 03 before 04, and 05 before 06/07.

Notebooks are committed without outputs. Run them after downloading the dataset to reproduce results.

**Note on `01_eda.ipynb`:** This notebook operates on raw, unlabeled race photos stored in `data/raw/`. These were scraped from public Flickr race albums and are not included in the repo. To run it, populate `data/raw/` with any collection of race photos. The EDA methodology (CLIP embeddings, UMAP, perceptual hashing) is dataset-agnostic. Notebooks 03–07 do not depend on `data/raw/` and can be run without it.

## Project Structure

```
race-vision/
  data/
    raw/          ← unlabeled images for EDA (not included — bring your own)
    labeled/      ← Roboflow export goes here
  notebooks/
    01_eda.ipynb             ← EDA on raw images
    03_training.ipynb        ← YOLOv8n fine-tuning
    04_evaluation.ipynb      ← YOLOv8 evaluation
    05_rfdetr_training.ipynb ← RF-DETR training
    06_rfdetr_evaluation.ipynb
    07_image_analysis.ipynb  ← image attribute analysis
  models/
    runs/         ← training output (weights, metrics) saved here
  src/
    predict.py    ← inference pipeline
```

## Key Tools

| Tool | Purpose |
|---|---|
| Ultralytics (YOLOv8/YOLO11) | Detection model training & inference |
| Roboflow | Labeling, dataset management, auto-label, health check |
| Apple MPS / CUDA | Local GPU — MPS on Apple Silicon, CUDA on Linux/Windows |
| CLIP | Image embeddings for diversity analysis |
| UMAP | Dimensionality reduction for diversity visualization |
| imagehash | Near-duplicate detection |
| EasyOCR | Bib number reading |
| Weights & Biases | Experiment tracking |
| torchvision | Attribute classifier training |

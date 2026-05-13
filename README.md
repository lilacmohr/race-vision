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

## Project Structure

```
race-vision/
  data/
    raw/          ← everything downloaded, unprocessed
    processed/    ← deduplicated, filtered
    labeled/      ← Roboflow export goes here
  notebooks/
    01_eda.ipynb
    02_label_audit.ipynb
    03_training.ipynb
    04_evaluation.ipynb
  models/
    runs/         ← Ultralytics saves training runs here
  src/
    pipeline.py   ← inference pipeline
    diversity.py  ← EDA utilities
```

## Key Tools

| Tool | Purpose |
|---|---|
| Ultralytics (YOLOv8/YOLO11) | Detection model training & inference |
| Roboflow | Labeling, dataset management, auto-label, health check |
| Apple MPS | Local GPU on M4 MacBook — no cloud needed |
| CLIP | Image embeddings for diversity analysis |
| UMAP | Dimensionality reduction for diversity visualization |
| imagehash | Near-duplicate detection |
| EasyOCR | Bib number reading |
| Weights & Biases | Experiment tracking |
| torchvision | Attribute classifier training |

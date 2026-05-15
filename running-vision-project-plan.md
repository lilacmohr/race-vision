# Running Gear Vision Model — Project Plan

## Phase 1: Environment Setup

- Install Ultralytics (YOLOv8/YOLO11), Python dependencies
- Set up local environment on Apple Silicon (MPS) — no CUDA needed
- Install EDA libraries: `clip`, `umap-learn`, `imagehash`, `matplotlib`, `pandas`
- Create a Roboflow account (free tier)
- Familiarize yourself with YOLO inference using a pretrained model on a few test images

---

## Phase 2: Dataset Collection (Raw)

- Search Roboflow Universe for existing running/sports datasets — pull multiple sources
- Scrape finish line and race photos from varied public sources (Flickr, SmugMug race albums, different races, geographies, years)
- Keep metadata per image: source dataset, race name, date — you'll use this to audit diversity
- Do **not** label yet — EDA comes first

---

## Phase 3: Exploratory Data Analysis (EDA)

This step separates thoughtful ML engineering from just running scripts. Understand your data before investing labeling time.

**Duplicate detection**
- Run perceptual hash (`imagehash`) across all images to find near-duplicates from race photo bursts
- Remove redundant images — they waste labeling effort and bias the model

**Visual inspection**
- Manually review a random sample of 50–100 images
- Flag homogeneity: same race font/colors, same camera angle, same lighting conditions

**Embedding-based diversity analysis**
- Run all images through CLIP to generate embeddings
- Reduce to 2D with UMAP and scatter plot, colored by source dataset
- Tight clusters mapping to a single source = homogenous data, go collect more variety

**Image statistics**
- Plot distribution of brightness, contrast, and resolution across the dataset
- Low standard deviation in brightness = not enough lighting variety

**Intentional composition decision**
- After EDA, make conscious sourcing decisions before labeling:
  - "I need trail and track races, not just road"
  - "I need mid-race shots, not just finish line"
  - "I need overcast and night race photos"
- Collect additional images to fill identified gaps, then re-run EDA

**Project structure to set up now:**
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

---

## Phase 4: Labeling (Detection Classes)

**Start with one class: `race_bib`**
- Visually distinctive, consistent appearance, easiest to learn the workflow with
- Add `shoe` as your second class once the pipeline is working end to end
- Add remaining classes one or two at a time, retraining each time

Use Roboflow's web labeler for bounding boxes. Full target class list for later:

- `race_bib`
- `shoe`
- `top` (singlet, shirt, jacket)
- `shorts` / `tights`
- `gps_watch`
- `sunglasses`
- `cap` / `visor`
- `hydration_vest`

**Labeling strategy:**
- Use Roboflow's auto-label (SAM2-backed) for a first pass on scraped images
- Human-review and correct every auto-label — do not trust them blindly
- Target ~150 quality-reviewed instances per class before training
- Run Roboflow's Health Check before every export (catches duplicates, imbalance, null labels)
- Split train / val / test (80 / 10 / 10) — ensure all three splits have varied sources, not the same race

**After labeling, audit with a notebook:**
- Visualize random samples from train and val sets
- Confirm label consistency (tight boxes, correct class assignments)
- Check that val/test images come from different sources than training images

---

## Phase 5: Train the Detection Model (YOLO)

- Export dataset from Roboflow in YOLO format
- Fine-tune `yolov8n` (nano) first — fast iteration, good for learning the workflow
- Track runs with Weights & Biases or Ultralytics' built-in logging
- Evaluate with **mAP50** and **mAP50-95**; inspect failure cases visually

**The real training loop:**
```
train → evaluate mAP → visually inspect failures →
diagnose: is it a data problem or model problem? →
almost always: get more/better data for weak cases → retrain
```

- When a class performs poorly, the fix is almost always more diverse images for that class — not hyperparameter tuning
- Once satisfied with nano, train `yolov8s` or `yolov8m` for better accuracy

---

## Phase 6: Bib Number Reading (OCR Extension)

Add a second stage to the bib detection pipeline:

- Crop detected bib bounding boxes from the original image (with ~15px padding)
- Pass crops to EasyOCR to extract the race number as a string
- Full pipeline: `image → YOLO → bib crop → EasyOCR → "4521"`
- This two-stage pattern (detect → read/classify the crop) is directly transferable to decorated apparel

---

## Phase 7: Attribute Classifier (Brand / Color)

- For each detection class (e.g., `shoe`, `top`), crop bounding box regions
- Label crops with attributes: brand (Nike, Brooks, HOKA, On, Asics, etc.) and optionally color
- Train a simple image classifier on the crops — ResNet18 or EfficientNet-B0 via `torchvision`
- Pipeline: YOLO detects → crop region → classifier predicts attributes
- This is the two-stage architecture used in production apparel systems

---

## Phase 8: Inference Pipeline

- Write a script that accepts an image and returns structured output:
  - Detected objects with bounding boxes
  - Attribute labels per detection (brand, color)
  - Confidence scores
- Visualize results with annotated bounding boxes (Ultralytics has built-in utilities)
- Test on fresh race photos not seen during training

---

## Key Tools & Resources

| Tool | Purpose |
|---|---|
| Ultralytics (YOLOv8/YOLO11) | Detection model training & inference |
| Roboflow | Labeling, dataset management, auto-label, health check |
| Apple MPS | Local GPU on M4 MacBook — no cloud needed |
| CLIP | Image embeddings for diversity analysis |
| UMAP | Dimensionality reduction for diversity visualization |
| imagehash | Near-duplicate detection |
| EasyOCR | Bib number reading |
| Weights & Biases | Experiment tracking (optional but useful) |
| torchvision | Attribute classifier training |
| MediaPipe | Pose estimation (if you add form analysis later) |

---

## Rough Timeline

| Phase | Estimated Effort |
|---|---|
| Setup | 1–2 hours |
| Data collection (raw) | 3–5 hours |
| EDA + diversity analysis | 3–5 hours |
| Labeling (first class) | 2–3 hours |
| First training run + evaluation | 2–4 hours |
| Iteration (data + retrain per class) | ongoing |
| OCR extension | 2–3 hours |
| Attribute classifier | 4–6 hours |
| Inference pipeline | 2–3 hours |

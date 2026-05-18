"""Run bib detection on a folder of images using RF-DETR Nano and save annotated results."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rfdetr import RFDETRNano

try:
    import supervision as sv
    HAS_SV = True
except ImportError:
    HAS_SV = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = PROJECT_ROOT / "models/runs/rfdetr-nano-map50/checkpoint_best_ema.pth"
DEFAULT_SOURCE  = PROJECT_ROOT / "data/personal"
DEFAULT_OUTPUT  = PROJECT_ROOT / "runs/rfdetr-detect"


def annotate_and_save(img: Image.Image, detections, out_path: Path) -> int:
    """Draw bounding boxes on img and write to out_path. Returns detection count."""
    img_np = np.array(img)

    if HAS_SV and len(detections) > 0:
        labels = [f"bib {c:.2f}" for c in detections.confidence]
        color = sv.Color.from_hex("#00FF00")
        annotator = sv.BoxAnnotator(color=color, thickness=4)
        label_annotator = sv.LabelAnnotator(color=color, text_color=sv.Color.BLACK)
        img_np = annotator.annotate(scene=img_np, detections=detections)
        img_np = label_annotator.annotate(scene=img_np, detections=detections, labels=labels)
    elif not HAS_SV and len(detections) > 0:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        for box, conf in zip(detections.xyxy, detections.confidence):
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, max(0, y1 - 14)), f"bib {conf:.2f}", fill="red")
        img_np = np.array(img)

    Image.fromarray(img_np).save(out_path)
    return len(detections)


def main():
    parser = argparse.ArgumentParser(description="RF-DETR Nano bib detector inference")
    parser.add_argument("--source",  default=str(DEFAULT_SOURCE),  help="Folder of input images")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to checkpoint .pth")
    parser.add_argument("--conf",    default=0.5, type=float,      help="Confidence threshold")
    parser.add_argument("--name",    default="personal",           help="Output subfolder name")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Source:  {args.source}")
    print(f"Weights: {args.weights}")

    model = RFDETRNano(pretrain_weights=args.weights)

    source = Path(args.source)
    out_dir = Path(DEFAULT_OUTPUT) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}
    images = [p for p in sorted(source.iterdir()) if p.suffix.lower() in exts]
    if not images:
        print(f"No images found in {source}")
        return

    total_detections = 0
    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        detections = model.predict(img, threshold=args.conf)
        count = annotate_and_save(img, detections, out_dir / img_path.name)
        total_detections += count
        print(f"  {img_path.name}: {count} bib(s) detected")

    print(f"\nDone — {len(images)} images processed, {total_detections} total detections.")
    print(f"Annotated images: {out_dir}")


if __name__ == "__main__":
    main()

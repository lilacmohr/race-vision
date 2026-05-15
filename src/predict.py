"""Run bib detection on a folder of images and save annotated results."""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = PROJECT_ROOT / "models/runs/bib-yolov8n/weights/best.pt"
DEFAULT_SOURCE  = PROJECT_ROOT / "data/personal"
DEFAULT_OUTPUT  = PROJECT_ROOT / "runs/detect"


def main():
    parser = argparse.ArgumentParser(description="Bib detector inference")
    parser.add_argument("--source",  default=str(DEFAULT_SOURCE),  help="Folder of input images")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to best.pt")
    parser.add_argument("--conf",    default=0.25, type=float,     help="Confidence threshold")
    parser.add_argument("--name",    default="personal",           help="Output subfolder name")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Source:  {args.source}")
    print(f"Weights: {args.weights}")

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        device=device,
        save=True,
        save_txt=True,
        project=str(DEFAULT_OUTPUT),
        name=args.name,
    )

    print(f"\nDone — {len(results)} images processed.")
    print(f"Annotated images: {DEFAULT_OUTPUT / args.name}")
    print(f"Label files:      {DEFAULT_OUTPUT / args.name / 'labels'}")


if __name__ == "__main__":
    main()

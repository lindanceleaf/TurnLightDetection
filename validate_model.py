from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    default_data = root / "yolo_dataset_day" / "yolo_dataset_day" / "data.yaml"
    default_model = root / "runs" / "turn_signal_day" / "yolov8s_img960" / "weights" / "best.pt"

    parser = argparse.ArgumentParser(description="Validate a trained YOLO turn-signal model.")
    parser.add_argument("--model", type=Path, default=default_model, help="Path to best.pt.")
    parser.add_argument("--data", type=Path, default=default_data, help="YOLO data.yaml path.")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--conf", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is not installed. Run: pip install -r requirements.txt") from exc

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    model = YOLO(str(args.model))
    metrics = model.val(
        data=str(args.data.resolve()),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
        conf=args.conf,
        plots=True,
    )
    print(metrics)


if __name__ == "__main__":
    main()

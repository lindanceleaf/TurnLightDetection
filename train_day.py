from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    default_data = root / "yolo_dataset_day" / "yolo_dataset_day" / "data.yaml"

    parser = argparse.ArgumentParser(
        description="Train a YOLO detector for CCTV turn-signal classification."
    )
    parser.add_argument("--data", type=Path, default=default_data, help="YOLO data.yaml path.")
    parser.add_argument(
        "--model",
        default="yolov8s.pt",
        help="Pretrained YOLO checkpoint, e.g. yolov8n.pt, yolov8s.pt, yolov8m.pt.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument(
        "--batch",
        default="8",
        help="Batch size. Use an integer like 8/12/16, or -1 for Ultralytics auto-batch.",
    )
    parser.add_argument("--device", default="0", help="CUDA device id. Use 0 for RTX 3080.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", default="runs/turn_signal_day")
    parser.add_argument("--name", default="yolov8s_img960")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM if you have enough memory.")
    parser.add_argument("--resume", action="store_true", help="Resume the previous run.")
    return parser.parse_args()


def normalize_batch(value: str) -> int:
    if value == "-1":
        return -1
    return int(value)


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Install dependencies first:\n"
            "  pip install -r requirements.txt\n"
            "If torch install fails on Python 3.13, create a Python 3.11 env first."
        ) from exc

    data_path = args.data.resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=normalize_batch(args.batch),
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        seed=args.seed,
        cache=args.cache,
        resume=args.resume,
        pretrained=True,
        optimizer="auto",
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        plots=True,
        val=True,
    )


if __name__ == "__main__":
    main()

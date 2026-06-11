from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    default_model = root / "runs" / "turn_signal_day" / "yolov8s_img960" / "weights" / "best.pt"

    parser = argparse.ArgumentParser(
        description="Run YOLO prediction or tracking on CCTV images/videos."
    )
    parser.add_argument("--source", required=True, help="Image, folder, video, RTSP URL, or webcam id.")
    parser.add_argument("--model", type=Path, default=default_model, help="Path to best.pt.")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--track", action="store_true", help="Use ByteTrack for video tracking.")
    parser.add_argument("--show", action="store_true", help="Open a display window.")
    parser.add_argument("--save-crops", action="store_true", help="Save cropped detections.")
    parser.add_argument("--project", default="runs/predict_turn_signal")
    parser.add_argument("--name", default="result")
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
    common = dict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        show=args.show,
        save=True,
        save_crop=args.save_crops,
        project=args.project,
        name=args.name,
    )

    if args.track:
        model.track(
            **common,
            tracker="bytetrack.yaml",
            persist=True,
            stream=False,
        )
    else:
        model.predict(**common)


if __name__ == "__main__":
    main()

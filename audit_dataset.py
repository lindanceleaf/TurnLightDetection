from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    default_data = root / "yolo_dataset_day" / "yolo_dataset_day" / "data.yaml"
    parser = argparse.ArgumentParser(description="Audit a YOLO detection dataset.")
    parser.add_argument("--data", type=Path, default=default_data)
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        return load_simple_data_yaml(path)

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_simple_data_yaml(path: Path) -> dict:
    cfg: dict = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "nc":
            cfg[key] = int(value)
        elif key == "names":
            value = value.strip("[]")
            cfg[key] = [item.strip().strip("'\"") for item in value.split(",") if item.strip()]
        else:
            cfg[key] = value.strip("'\"")
    return cfg


def resolve_split(root: Path, split_value: str) -> Path:
    split_path = Path(split_value)
    if split_path.is_absolute():
        return split_path
    return root / split_path


def label_path_for(image_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    relative = image_path.relative_to(images_dir)
    return labels_dir / relative.with_suffix(".txt")


def read_label_file(path: Path) -> tuple[Counter, list[str]]:
    counts: Counter = Counter()
    errors: list[str] = []
    if not path.exists():
        return counts, [f"missing label: {path}"]

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path}:{line_no} expected 5 columns, got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            xywh = [float(v) for v in parts[1:]]
        except ValueError:
            errors.append(f"{path}:{line_no} non-numeric label")
            continue
        if cls < 0:
            errors.append(f"{path}:{line_no} negative class id")
        if any(v < 0 or v > 1 for v in xywh):
            errors.append(f"{path}:{line_no} xywh outside 0..1")
        if xywh[2] <= 0 or xywh[3] <= 0:
            errors.append(f"{path}:{line_no} width/height must be positive")
        counts[cls] += 1
    return counts, errors


def clip_id(path: Path) -> str:
    match = re.search(r"clip_\d+", path.stem)
    return match.group(0) if match else "unknown"


def labels_dir_from_images_dir(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            parts[idx] = "labels"
            return Path(*parts)
    raise ValueError(f"Cannot infer labels directory from {images_dir}")


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.data.resolve())
    root = Path(cfg["path"]).expanduser()
    if not root.is_absolute():
        root = (args.data.parent / root).resolve()

    names = cfg.get("names", [])
    splits = [s for s in ("train", "val", "test") if s in cfg]

    print(f"dataset root: {root}")
    print(f"classes: {names}")
    print()

    all_errors: list[str] = []
    split_clips: dict[str, set[str]] = {}

    for split in splits:
        images_dir = resolve_split(root, cfg[split]).resolve()
        labels_dir = labels_dir_from_images_dir(images_dir)
        images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)

        class_counts: Counter = Counter()
        clip_counts: defaultdict[str, int] = defaultdict(int)
        for image in images:
            clip_counts[clip_id(image)] += 1
            label_path = label_path_for(image, images_dir, labels_dir)
            counts, errors = read_label_file(label_path)
            class_counts.update(counts)
            if len(all_errors) < 20:
                all_errors.extend(errors[: 20 - len(all_errors)])

        split_clips[split] = set(clip_counts)
        print(f"[{split}]")
        print(f"images: {len(images)}")
        print(f"clips: {len(clip_counts)}")
        print("objects:")
        for idx, name in enumerate(names):
            print(f"  {idx} {name}: {class_counts.get(idx, 0)}")
        unknown = sorted(k for k in class_counts if k >= len(names))
        if unknown:
            print(f"  unknown class ids: {unknown}")
        print()

    if "train" in split_clips and "val" in split_clips:
        overlap = split_clips["train"] & split_clips["val"]
        print(f"train/val clip overlap: {len(overlap)}")
        if overlap:
            print("overlapped clips:", ", ".join(sorted(overlap)[:20]))

    if all_errors:
        print()
        print("label issues (first 20):")
        for error in all_errors[:20]:
            print(f"  {error}")


if __name__ == "__main__":
    main()

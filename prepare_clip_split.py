from __future__ import annotations

import argparse
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from audit_dataset import IMAGE_EXTS, clip_id, labels_dir_from_images_dir, load_yaml


@dataclass
class Sample:
    image: Path
    label: Path
    clip: str


@dataclass
class ClipGroup:
    name: str
    samples: list[Sample] = field(default_factory=list)
    class_counts: Counter = field(default_factory=Counter)

    @property
    def image_count(self) -> int:
        return len(self.samples)

    @property
    def object_count(self) -> int:
        return sum(self.class_counts.values())


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    default_data = root / "yolo_dataset_day" / "yolo_dataset_day" / "data.yaml"
    default_out = root / "yolo_dataset_day_clip_split"

    parser = argparse.ArgumentParser(
        description="Create a new YOLO dataset split by clip id to avoid train/val leakage."
    )
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating hard links.",
    )
    return parser.parse_args()


def dataset_root(data_yaml: Path, cfg: dict) -> Path:
    root = Path(cfg["path"]).expanduser()
    if root.is_absolute():
        return root.resolve()
    return (data_yaml.parent / root).resolve()


def read_class_counts(label_path: Path) -> Counter:
    counts: Counter = Counter()
    if not label_path.exists():
        return counts
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) != 5:
            continue
        try:
            counts[int(parts[0])] += 1
        except ValueError:
            continue
    return counts


def collect_samples(root: Path, cfg: dict) -> dict[str, ClipGroup]:
    groups: dict[str, ClipGroup] = {}
    seen: set[str] = set()

    for split in ("train", "val", "test"):
        if split not in cfg:
            continue
        images_dir = (root / cfg[split]).resolve()
        labels_dir = labels_dir_from_images_dir(images_dir)
        if not images_dir.exists():
            continue

        for image in sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
            key = image.stem
            if key in seen:
                continue
            seen.add(key)

            label = labels_dir / image.relative_to(images_dir).with_suffix(".txt")
            group_name = clip_id(image)
            group = groups.setdefault(group_name, ClipGroup(group_name))
            sample = Sample(image=image, label=label, clip=group_name)
            group.samples.append(sample)
            group.class_counts.update(read_class_counts(label))

    return groups


def distance(counts: Counter, target_counts: Counter, image_count: int, target_images: int) -> float:
    class_keys = set(counts) | set(target_counts)
    class_cost = 0.0
    for key in class_keys:
        target = max(target_counts[key], 1)
        class_cost += abs(counts[key] - target_counts[key]) / target
    image_target = max(target_images, 1)
    image_cost = abs(image_count - target_images) / image_target
    return class_cost + image_cost


def select_split(
    available: dict[str, ClipGroup],
    ratio: float,
    total_counts: Counter,
    total_images: int,
) -> set[str]:
    if ratio <= 0 or not available:
        return set()

    target_counts = Counter({key: round(value * ratio) for key, value in total_counts.items()})
    target_images = max(1, round(total_images * ratio))
    selected: set[str] = set()
    current_counts: Counter = Counter()
    current_images = 0

    while current_images < target_images and len(selected) < len(available):
        best_name: str | None = None
        best_score: float | None = None
        for name, group in available.items():
            if name in selected:
                continue
            candidate_counts = current_counts + group.class_counts
            candidate_images = current_images + group.image_count
            score = distance(candidate_counts, target_counts, candidate_images, target_images)
            if best_score is None or score < best_score:
                best_score = score
                best_name = name
        if best_name is None:
            break
        selected.add(best_name)
        current_counts.update(available[best_name].class_counts)
        current_images += available[best_name].image_count

    return selected


def make_link_or_copy(source: Path, destination: Path, copy: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_sample(sample: Sample, output_root: Path, split: str, copy: bool) -> None:
    image_dst = output_root / "images" / split / sample.image.name
    label_dst = output_root / "labels" / split / sample.label.name
    make_link_or_copy(sample.image, image_dst, copy)
    if sample.label.exists():
        make_link_or_copy(sample.label, label_dst, copy)
    else:
        label_dst.parent.mkdir(parents=True, exist_ok=True)
        label_dst.write_text("", encoding="utf-8")


def write_data_yaml(output_root: Path, names: list[str], has_test: bool) -> None:
    lines = [
        f"path: {output_root.as_posix()}",
        "train: images/train",
        "val: images/val",
    ]
    if has_test:
        lines.append("test: images/test")
    lines.extend(
        [
            f"nc: {len(names)}",
            f"names: {names!r}",
            "",
        ]
    )
    (output_root / "data.yaml").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("Require 0 <= val_ratio, 0 <= test_ratio, and val_ratio + test_ratio < 1.")

    cfg = load_yaml(args.data.resolve())
    root = dataset_root(args.data.resolve(), cfg)
    names = cfg.get("names", [])
    groups = collect_samples(root, cfg)
    if not groups:
        raise RuntimeError(f"No samples found under {root}")

    output_root = args.out.resolve()
    workspace = Path(__file__).resolve().parent
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists. Add --overwrite to replace it.")
        if workspace not in output_root.parents and output_root != workspace:
            raise ValueError(f"Refusing to overwrite outside workspace: {output_root}")
        shutil.rmtree(output_root)

    rng = random.Random(args.seed)
    shuffled_names = list(groups)
    rng.shuffle(shuffled_names)
    shuffled_groups = {name: groups[name] for name in shuffled_names}

    total_counts: Counter = Counter()
    total_images = 0
    for group in groups.values():
        total_counts.update(group.class_counts)
        total_images += group.image_count

    test_names = select_split(shuffled_groups, args.test_ratio, total_counts, total_images)
    remaining = {name: group for name, group in shuffled_groups.items() if name not in test_names}
    val_ratio_adjusted = args.val_ratio / (1 - args.test_ratio)
    val_names = select_split(remaining, val_ratio_adjusted, total_counts, total_images)
    train_names = set(groups) - val_names - test_names

    assignments = {"train": train_names, "val": val_names}
    if test_names:
        assignments["test"] = test_names

    for split, names_for_split in assignments.items():
        for name in sorted(names_for_split):
            for sample in groups[name].samples:
                write_sample(sample, output_root, split, args.copy)

    write_data_yaml(output_root, names, bool(test_names))

    print(f"created: {output_root}")
    for split, names_for_split in assignments.items():
        class_counts: Counter = Counter()
        image_count = 0
        for name in names_for_split:
            class_counts.update(groups[name].class_counts)
            image_count += groups[name].image_count
        print(f"[{split}] clips={len(names_for_split)} images={image_count}")
        for idx, class_name in enumerate(names):
            print(f"  {idx} {class_name}: {class_counts[idx]}")


if __name__ == "__main__":
    main()

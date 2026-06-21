"""Prepare COCO detection dataset for EdgeLocate training in JSONL format.

Each image becomes one sample: all object boxes are listed as
<box><d1><d2><d3><d4></box> tokens in the assistant response.

Usage:
    python -c "from locany.prepare_coco import prepare; prepare()"

Or via train.py:
    python train.py --action prepare_coco
"""

import json
import os
import subprocess
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from .utils import boxes_to_tokens, logger, SPECIAL_TOKENS

COCO_IMAGE_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "train2014": "http://images.cocodataset.org/zips/train2014.zip",
    "val2014": "http://images.cocodataset.org/zips/val2014.zip",
}

COCO_ANN_URLS = {
    "2017": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "2014": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
}

COCO_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def download_coco_images(coco_root: str, year: str = "2017"):
    """Download COCO images if not present."""
    split_map = {"2017": ("train2017", "val2017"), "2014": ("train2014", "val2014")}
    for split in split_map[year]:
        target_dir = os.path.join(coco_root, split)
        if os.path.isdir(target_dir) and len(os.listdir(target_dir)) > 100:
            logger.info(f"COCO {split} already exists")
            continue
        url = COCO_IMAGE_URLS[split]
        zip_path = os.path.join(coco_root, f"{split}.zip")
        os.makedirs(coco_root, exist_ok=True)
        logger.info(f"Downloading COCO {split}...")
        subprocess.run(
            ["wget", "-O", zip_path, url],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["unzip", "-q", zip_path, "-d", coco_root],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.remove(zip_path)
        logger.info(f"COCO {split} ready at {target_dir}")


def download_coco_annotations(coco_root: str, year: str = "2017") -> str:
    """Download COCO annotations, returns path to annotation dir."""
    ann_dir = os.path.join(coco_root, "annotations")
    if os.path.isdir(ann_dir) and len(os.listdir(ann_dir)) > 3:
        logger.info(f"COCO annotations already exist at {ann_dir}")
        return ann_dir

    url = COCO_ANN_URLS[year]
    zip_path = os.path.join(coco_root, "annotations.zip")
    logger.info("Downloading COCO annotations...")
    subprocess.run(
        ["wget", "-O", zip_path, url],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["unzip", "-q", zip_path, "-d", coco_root],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.remove(zip_path)
    logger.info(f"COCO annotations ready at {ann_dir}")
    return ann_dir


def build_category_map(ann_data: Dict) -> Dict[int, str]:
    """Build mapping from category_id to category name."""
    return {cat["id"]: cat["name"] for cat in ann_data.get("categories", [])}


def generate_prompts(num_prompts: int = 20) -> List[str]:
    """Generate diverse detection prompts."""
    base = [
        "Detect all objects in this image.",
        "Find all objects in this image.",
        "Locate every object in this image.",
        "List all objects with their bounding boxes.",
        "Detect every visible object.",
    ]
    specific = [
        f"Detect all {cat} in this image."
        for cat in COCO_CATEGORIES[:15]
    ]
    return base + specific


def convert_coco_to_jsonl(
    ann_path: str,
    coco_root: str,
    output_path: str,
    split: str = "train",
    year: str = "2017",
    max_boxes_per_image: int = 50,
    min_box_area: int = 400,
    max_images: Optional[int] = None,
    prompt_template: str = "Detect all objects in this image.",
):
    """Convert COCO detection annotations to JSONL format.

    Each sample: user prompt → assistant response with <box> tokens
    for every object in the image (up to max_boxes_per_image).

    Args:
        ann_path: Path to instances_{split}{year}.json
        coco_root: Root dir containing COCO_{split} image dirs
        output_path: Where to write JSONL
        split: train/val
        year: 2017 or 2014
        max_boxes_per_image: Cap on boxes per sample
        min_box_area: Minimum area to include a box
        max_images: Limit total images processed
        prompt_template: Prompt to use (can include {categories})
    """
    logger.info(f"Loading COCO annotations from {ann_path}")
    with open(ann_path) as f:
        ann_data = json.load(f)

    cat_map = build_category_map(ann_data)

    # Index annotations by image_id
    img_to_anns = defaultdict(list)
    for ann in ann_data.get("annotations", []):
        img_to_anns[ann["image_id"]].append(ann)

    # Filter to split images if split field exists
    images = ann_data.get("images", [])
    if split == "train":
        images = [img for img in images if "val" not in img.get("file_name", "train")]
    elif split == "val":
        images = [img for img in images if "val" in img.get("file_name", "")]

    if max_images:
        images = images[:max_images]

    image_subdir = f"{split}{year}"
    written = 0
    skipped = 0
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as out_f:
        for img in tqdm(images, desc=f"COCO {split} -> JSONL"):
            image_id = img["id"]
            fname = img.get("file_name", "")
            if not fname:
                fname = f"{image_id:012d}.jpg"

            img_path = os.path.join(coco_root, image_subdir, fname)
            if not os.path.exists(img_path):
                img_path = os.path.join(coco_root, fname)
            if not os.path.exists(img_path):
                skipped += 1
                continue

            anns = img_to_anns.get(image_id, [])
            boxes = []
            for ann in anns:
                bbox = ann.get("bbox", [])
                if len(bbox) < 4:
                    continue
                x, y, w, h = bbox[:4]
                if w * h < min_box_area:
                    continue
                img_w = img.get("width", 640)
                img_h = img.get("height", 480)
                x1 = int(max(0, x) * 1000 / img_w)
                y1 = int(max(0, y) * 1000 / img_h)
                x2 = int(min(img_w, x + w) * 1000 / img_w)
                y2 = int(min(img_h, y + h) * 1000 / img_h)
                x1 = min(1000, max(0, x1))
                y1 = min(1000, max(0, y1))
                x2 = min(1000, max(0, x2))
                y2 = min(1000, max(0, y2))
                boxes.append([x1, y1, x2, y2])

            if not boxes:
                skipped += 1
                continue

            if len(boxes) > max_boxes_per_image:
                boxes = boxes[:max_boxes_per_image]

            box_token_str = boxes_to_tokens(boxes)

            sample = {
                "image": img_path,
                "conversations": [
                    {
                        "from": "user",
                        "value": f"{SPECIAL_TOKENS['image']}\n{prompt_template}",
                    },
                    {
                        "from": "assistant",
                        "value": box_token_str,
                        "boxes": boxes,
                    },
                ],
            }

            out_f.write(json.dumps(sample) + "\n")
            written += 1

    logger.info(f"Wrote {written} samples to {output_path} (skipped {skipped})")
    return written


def prepare(
    coco_root: str = "./data/coco",
    output_dir: str = "./data/coco_detection",
    year: str = "2017",
    splits: Tuple[str, str] = ("train", "val"),
    max_images_per_split: Optional[Dict[str, int]] = None,
    max_boxes_per_image: int = 50,
    prompt_template: str = "Detect all objects in this image.",
    download: bool = True,
):
    """End-to-end: download COCO + convert to JSONL.

    Args:
        coco_root: Where COCO images live/will be downloaded
        output_dir: Where to write JSONL files
        year: 2017 or 2014
        splits: Which splits to process
        max_images_per_split: e.g. {"train": 50000, "val": 1000}
        max_boxes_per_image: Cap boxes per sample
        prompt_template: Prompt for every sample
        download: Auto-download missing data
    """
    os.makedirs(output_dir, exist_ok=True)

    if download:
        download_coco_images(coco_root, year=year)
        ann_dir = download_coco_annotations(coco_root, year=year)
    else:
        ann_dir = os.path.join(coco_root, "annotations")

    if max_images_per_split is None:
        max_images_per_split = {}

    for split in splits:
        ann_path = os.path.join(ann_dir, f"instances_{split}{year}.json")
        if not os.path.exists(ann_path):
            logger.warning(f"Annotations not found: {ann_path}")
            continue

        out_path = os.path.join(output_dir, f"{split}.jsonl")
        convert_coco_to_jsonl(
            ann_path=ann_path,
            coco_root=coco_root,
            output_path=out_path,
            split=split,
            year=year,
            max_boxes_per_image=max_boxes_per_image,
            max_images=max_images_per_split.get(split),
            prompt_template=prompt_template,
        )

    logger.info(f"Done. Files in {output_dir}/")
    for fname in sorted(os.listdir(output_dir)):
        if fname.endswith(".jsonl"):
            fpath = os.path.join(output_dir, fname)
            count = sum(1 for _ in open(fpath))
            logger.info(f"  {fname}: {count} samples ({os.path.getsize(fpath)/1e6:.1f} MB)")


def add_coco_parser(subparsers):
    """Add coco subcommand to argparse."""
    parser = subparsers.add_parser("prepare_coco", help="Prepare COCO detection dataset")
    parser.add_argument("--coco_root", default="./data/coco", help="COCO image directory")
    parser.add_argument("--output_dir", default="./data/coco_detection", help="Output directory")
    parser.add_argument("--year", default="2017", choices=["2014", "2017"], help="COCO year")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits")
    parser.add_argument("--max_train", type=int, default=None, help="Limit train images")
    parser.add_argument("--max_val", type=int, default=None, help="Limit val images")
    parser.add_argument("--max_boxes", type=int, default=50, help="Max boxes per image")
    parser.add_argument("--prompt", default="Detect all objects in this image.", help="Prompt template")
    parser.add_argument("--no-download", action="store_true", help="Skip download")
    return parser

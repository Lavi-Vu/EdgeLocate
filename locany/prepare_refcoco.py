"""Prepare RefCOCO+ dataset for EdgeLocate training in JSONL format.

Downloads COCO images and RefCOCO+ annotations, converts to sharegpt JSONL.

Usage:
    python -c "from locany.prepare_refcoco import prepare; prepare()"

Or with custom paths:
    python -c "from locany.prepare_refcoco import prepare; prepare(
        coco_root='/data/coco',
        output_dir='./data/refcoco_plus',
        ann_file='/data/annotations/refcoco+.json',
        split='train',
    )"
"""

import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from .utils import boxes_to_tokens, logger, set_seed, SPECIAL_TOKENS

COCO_URLS = {
    "train2014": "http://images.cocodataset.org/zips/train2014.zip",
    "val2014": "http://images.cocodataset.org/zips/val2014.zip",
}

REFCOCO_PLUS_URL = (
    "https://bobbywu.com/refcoco+/refcoco+.zip"
)


def download_coco(coco_root: str, years: List[str] = ("train2014", "val2014")):
    """Download COCO images if not present."""
    for year in years:
        target_dir = os.path.join(coco_root, year)
        if os.path.isdir(target_dir) and len(os.listdir(target_dir)) > 100:
            logger.info(f"COCO {year} already exists at {target_dir}")
            continue
        logger.info(f"Downloading COCO {year}...")
        url = COCO_URLS[year]
        zip_path = os.path.join(coco_root, f"{year}.zip")
        os.makedirs(coco_root, exist_ok=True)
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
        logger.info(f"COCO {year} downloaded to {target_dir}")


def download_refcoco_annotations(ann_dir: str) -> str:
    """Download RefCOCO+ annotations, returns path to annotations JSON."""
    os.makedirs(ann_dir, exist_ok=True)
    ann_path = os.path.join(ann_dir, "refcoco+.json")
    if os.path.exists(ann_path):
        logger.info(f"RefCOCO+ annotations already at {ann_path}")
        return ann_path

    zip_path = os.path.join(ann_dir, "refcoco+.zip")
    logger.info("Downloading RefCOCO+ annotations...")
    try:
        subprocess.run(
            ["wget", "-O", zip_path, REFCOCO_PLUS_URL],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["unzip", "-q", "-o", zip_path, "-d", ann_dir],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.remove(zip_path)

        extracted = [f for f in os.listdir(ann_dir) if f.endswith(".json")]
        if extracted:
            return os.path.join(ann_dir, extracted[0])
    except Exception as e:
        logger.warning(f"Download failed: {e}")
        logger.info("Please download manually from https://github.com/lichengunc/refer")

    return ann_path if os.path.exists(ann_path) else ""


def convert_refcoco_to_jsonl(
    ann_path: str,
    coco_root: str,
    output_path: str,
    split: str = "train",
    min_box_size: int = 5,
    max_boxes_per_sample: int = 1,
    num_samples: Optional[int] = None,
) -> int:
    """Convert RefCOCO+ annotations to JSONL format.

    Each sample: user prompt = expression, assistant response = <box><d><d><d><d></box>

    Returns number of samples written.
    """
    logger.info(f"Loading annotations from {ann_path}")

    with open(ann_path) as f:
        refcoco_data = json.load(f)

    images_info = {img["id"]: img for img in refcoco_data.get("images", [])}
    annotations_info = {ann["id"]: ann for ann in refcoco_data.get("annotations", [])}

    refs = [r for r in refcoco_data.get("annotations", []) if r.get("split") == split]
    if not refs:
        refs = [r for r in refcoco_data.get("annotations", [])]

    if num_samples:
        refs = refs[:num_samples]

    written = 0
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as out_f:
        for ref in tqdm(refs, desc=f"Converting RefCOCO+ ({split})"):
            image_id = ref.get("image_id") or ref.get("id")
            img_info = images_info.get(image_id)
            if not img_info:
                continue

            coco_split = {"train": "train2014", "val": "val2014", "test": "val2014"}.get(
                img_info.get("split", split), "train2014"
            )

            if "file_name" in img_info:
                fname = img_info["file_name"]
            else:
                fname = f"COCO_{coco_split}_{image_id:012d}.jpg"

            img_path = os.path.join(coco_root, coco_split, fname)
            if not os.path.exists(img_path):
                alt_path = os.path.join(coco_root, coco_split, fname)
                if not os.path.exists(alt_path):
                    continue
                img_path = alt_path

            sentences = ref.get("sentences", [ref])
            for sent in sentences:
                if isinstance(sent, dict):
                    expression = sent.get("raw") or sent.get("sentence", "")
                else:
                    expression = str(sent)

                if not expression.strip():
                    continue

                bbox = ref.get("bbox", ref.get("box", []))
                if len(bbox) < 4:
                    continue

                x, y, w_box, h_box = bbox[:4]
                if w_box < min_box_size or h_box < min_box_size:
                    continue

                x1_norm = int(x * 1000 / img_info.get("width", 640))
                y1_norm = int(y * 1000 / img_info.get("height", 480))
                x2_norm = int((x + w_box) * 1000 / img_info.get("width", 640))
                y2_norm = int((y + h_box) * 1000 / img_info.get("height", 480))

                x1_norm = max(0, min(1000, x1_norm))
                y1_norm = max(0, min(1000, y1_norm))
                x2_norm = max(0, min(1000, x2_norm))
                y2_norm = max(0, min(1000, y2_norm))

                box_token_str = boxes_to_tokens([[x1_norm, y1_norm, x2_norm, y2_norm]])

                sample = {
                    "image": img_path,
                    "conversations": [
                        {
                            "from": "user",
                            "value": f"{SPECIAL_TOKENS['image']}\n{expression}",
                        },
                        {
                            "from": "assistant",
                            "value": box_token_str,
                            "boxes": [[x1_norm, y1_norm, x2_norm, y2_norm]],
                        },
                    ],
                }

                out_f.write(json.dumps(sample) + "\n")
                written += 1

    logger.info(f"Wrote {written} samples to {output_path}")
    return written


def prepare(
    coco_root: str = "./data/coco",
    ann_path: str = "",
    output_dir: str = "./data/refcoco_plus",
    splits: List[str] = ("train", "val"),
    num_train: Optional[int] = None,
    num_val: Optional[int] = None,
    download: bool = True,
):
    """End-to-end: download + convert + split.

    Args:
        coco_root: Path to COCO images (COCO_{split}/ subdirs expected)
        ann_path: Path to RefCOCO+ annotations JSON. If empty, downloads.
        output_dir: Where to write train.jsonl / val.jsonl
        splits: Which splits to process
        num_train: Limit train samples
        num_val: Limit val samples
        download: Whether to auto-download missing data
    """
    os.makedirs(output_dir, exist_ok=True)

    if download and not ann_path:
        ann_dir = os.path.join(output_dir, "annotations")
        ann_path = download_refcoco_annotations(ann_dir)

    if not ann_path or not os.path.exists(ann_path):
        logger.error(
            "RefCOCO+ annotations not found. Please provide --ann_path or "
            "download manually from https://github.com/lichengunc/refer"
        )
        return

    if download:
        download_coco(coco_root)

    for split in splits:
        limit = {"train": num_train, "val": num_val}.get(split)
        out_path = os.path.join(output_dir, f"{split}.jsonl")
        convert_refcoco_to_jsonl(
            ann_path=ann_path,
            coco_root=coco_root,
            output_path=out_path,
            split=split,
            num_samples=limit,
        )

    logger.info(f"Done. Files in {output_dir}/")
    for fname in os.listdir(output_dir):
        if fname.endswith(".jsonl"):
            fpath = os.path.join(output_dir, fname)
            count = sum(1 for _ in open(fpath))
            logger.info(f"  {fname}: {count} samples")


def add_refcoco_parser(subparsers):
    """Add refcoco subcommand to argparse."""
    parser = subparsers.add_parser("prepare_refcoco", help="Prepare RefCOCO+ dataset")
    parser.add_argument("--coco_root", default="./data/coco", help="COCO image directory")
    parser.add_argument("--ann_path", default="", help="RefCOCO+ annotations JSON path")
    parser.add_argument("--output_dir", default="./data/refcoco_plus", help="Output directory")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to process")
    parser.add_argument("--num_train", type=int, default=None, help="Limit train samples")
    parser.add_argument("--num_val", type=int, default=None, help="Limit val samples")
    parser.add_argument("--no-download", action="store_true", help="Skip download")
    return parser

"""Prepare RefCOCO/RefCOCO+/RefCOCOg datasets for EdgeLocate training.

Downloads pre-processed annotations from PaDT-MLLM/RefCOCO on HuggingFace,
converts all three variants to JSONL, and combines them.

Usage:
    python -c "from locany.prepare_refcoco import prepare; prepare()"
"""

import json
import os
import subprocess
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from .utils import boxes_to_tokens, logger, SPECIAL_TOKENS

REPO_ID = "PaDT-MLLM/RefCOCO"

COCO_URLS = {
    "train2014": "http://images.cocodataset.org/zips/train2014.zip",
    "val2014": "http://images.cocodataset.org/zips/val2014.zip",
}

VARIANT_FILES = {
    "refcoco": {
        "train": "refcoco_train.json",
        "val": "refcoco_val.json",
        "testA": "refcoco_testA.json",
        "testB": "refcoco_testB.json",
    },
    "refcoco+": {
        "train": "refcoco+_train.json",
        "val": "refcoco+_val.json",
        "testA": "refcoco+_testA.json",
        "testB": "refcoco+_testB.json",
    },
    "refcocog": {
        "train": "refcocog_train.json",
        "val": "refcocog_val.json",
        "test": "refcocog_test.json",
    },
}


def download_hf_annotations(output_dir: str) -> str:
    """Download all annotation files from HF repo. Returns path to downloaded dir."""
    from huggingface_hub import hf_hub_download
    ann_dir = os.path.join(output_dir, "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    for variant, splits in VARIANT_FILES.items():
        for split, fname in splits.items():
            local = os.path.join(ann_dir, fname)
            if os.path.exists(local):
                logger.info(f"  {fname} already exists, skipping")
                continue
            logger.info(f"Downloading {fname} ...")
            hf_hub_download(
                REPO_ID, fname, repo_type="dataset", local_dir=ann_dir,
            )
    return ann_dir


def download_coco(coco_root: str, years: Tuple[str, ...] = ("train2014", "val2014")):
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


def _resolve_image(coco_root: str, image_file: str) -> Optional[str]:
    """Resolve image path. RefCOCO images are all in train2014 or val2014."""
    for year in ("train2014", "val2014"):
        path = os.path.join(coco_root, year, image_file)
        if os.path.exists(path):
            return path
    return None


def convert_file(
    input_path: str,
    coco_root: str,
    output_path: str,
    limit: Optional[int] = None,
) -> int:
    """Convert a single PaDT-MLLM annotation file to EdgeLocate JSONL format.

    Each input line: {image, conversations, objects: [{bbox, label, ...}]}
    Output: {image, conversations: [{from: user/assistant, value: ...}]}
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    written = 0
    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in tqdm(f_in, desc=f"Converting {os.path.basename(input_path)}"):
            if limit and written >= limit:
                break
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            image_file = entry.get("image", "")
            convs = entry.get("conversations", [])
            objects = entry.get("objects", [])

            if not image_file or not convs or not objects:
                continue

            user_text = ""
            for c in convs:
                if c.get("from") in ("human", "user"):
                    user_text = c.get("value", "")
                    break

            if not user_text:
                continue

            img_path = _resolve_image(coco_root, image_file)
            if img_path is None:
                continue

            assistant_parts = []
            for obj in objects:
                bbox = obj.get("bbox")
                label = obj.get("label", "")
                if not bbox or len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = bbox[:4]
                x1_norm = max(0, min(1000, int(x1 * 1000)))
                y1_norm = max(0, min(1000, int(y1 * 1000)))
                x2_norm = max(0, min(1000, int(x2 * 1000)))
                y2_norm = max(0, min(1000, int(y2 * 1000)))
                box_str = f"<{x1_norm}><{y1_norm}><{x2_norm}><{y2_norm}>"
                assistant_parts.append(f"<ref>{label}</ref><box>{box_str}</box>")

            if not assistant_parts:
                continue

            assistant_text = " ".join(assistant_parts)

            sample = {
                "image": img_path,
                "conversations": [
                    {"from": "user", "value": f"{SPECIAL_TOKENS['image']}\n{user_text}"},
                    {"from": "assistant", "value": assistant_text},
                ],
            }
            f_out.write(json.dumps(sample) + "\n")
            written += 1

    logger.info(f"Wrote {written} samples to {output_path}")
    return written


def combine_jsonls(input_paths: List[str], output_path: str):
    """Concatenate multiple JSONL files into one."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total = 0
    with open(output_path, "w") as f_out:
        for path in input_paths:
            if not os.path.exists(path):
                logger.warning(f"  {path} not found, skipping")
                continue
            count = 0
            with open(path) as f_in:
                for line in f_in:
                    if line.strip():
                        f_out.write(line)
                        count += 1
            logger.info(f"  {os.path.basename(path)}: {count} samples")
            total += count
    logger.info(f"Combined {total} samples into {output_path}")


def prepare(
    coco_root: str = "./data/coco",
    output_dir: str = "./data/refcoco",
    ann_dir: str = "",
    splits: Tuple[str, ...] = ("train", "val"),
    num_train: Optional[int] = None,
    num_val: Optional[int] = None,
    download: bool = True,
    combine: bool = True,
):
    """Download RefCOCO/+/g annotations from HF and convert to EdgeLocate JSONL.

    Args:
        coco_root: Path to COCO images (train2014/val2014 subdirs expected)
        output_dir: Where to write JSONL files
        ann_dir: Path to pre-downloaded annotation directory. If empty, downloads.
        splits: Which splits to process ('train', 'val', 'testA', 'testB', 'test')
        num_train: Limit train samples per variant
        num_val: Limit val samples per variant
        download: Whether to auto-download COCO images and annotations
        combine: Whether to merge all variants into single train.jsonl/val.jsonl
    """
    os.makedirs(output_dir, exist_ok=True)

    # Download annotations from HF
    if download or not ann_dir:
        ann_dir = download_hf_annotations(output_dir)
    elif not os.path.isdir(ann_dir):
        ann_dir = download_hf_annotations(output_dir)

    # Download COCO images
    if download:
        coco_years = set()
        if "train" in splits:
            coco_years.add("train2014")
        if any(s in splits for s in ("val", "testA", "testB", "test")):
            coco_years.add("val2014")
        download_coco(coco_root, tuple(coco_years))

    # Convert each variant
    converted = {s: [] for s in splits}
    for variant, split_map in VARIANT_FILES.items():
        for split in splits:
            fname = split_map.get(split)
            if not fname:
                continue
            ann_path = os.path.join(ann_dir, fname)
            if not os.path.exists(ann_path):
                logger.warning(f"  {fname} not found in {ann_dir}, skipping")
                continue

            limit = None
            if split == "train":
                limit = num_train
            elif split in ("val", "testA", "testB", "test"):
                limit = num_val

            out_path = os.path.join(output_dir, f"{variant}_{split}.jsonl")
            count = convert_file(ann_path, coco_root, out_path, limit=limit)
            if count > 0:
                converted[split].append(out_path)

    # Combine per-split
    if combine:
        for split in splits:
            if converted[split]:
                combined_path = os.path.join(output_dir, f"{split}.jsonl")
                combine_jsonls(converted[split], combined_path)

    logger.info(f"Done. Files in {output_dir}/")
    for fname in sorted(os.listdir(output_dir)):
        if fname.endswith(".jsonl"):
            fpath = os.path.join(output_dir, fname)
            count = sum(1 for _ in open(fpath))
            logger.info(f"  {fname}: {count} samples")


def add_refcoco_parser(subparsers):
    """Add refcoco subcommand to argparse."""
    parser = subparsers.add_parser("prepare_refcoco", help="Prepare RefCOCO/+/g dataset")
    parser.add_argument("--coco_root", default="./data/coco", help="COCO image directory")
    parser.add_argument("--ann_dir", default="", help="Path to pre-downloaded annotations")
    parser.add_argument("--output_dir", default="./data/refcoco", help="Output directory")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to process")
    parser.add_argument("--num_train", type=int, default=None, help="Limit train samples per variant")
    parser.add_argument("--num_val", type=int, default=None, help="Limit val samples per variant")
    parser.add_argument("--no-download", action="store_true", help="Skip download")
    parser.add_argument("--no-combine", action="store_true", help="Skip combining variants")
    return parser

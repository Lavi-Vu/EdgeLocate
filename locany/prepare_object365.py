"""Prepare Objects365 dataset for EdgeLocate training.

Downloads Object365 v2 images and annotations, converts to JSONL format.

Usage:
    python -c "from locany.prepare_object365 import prepare; prepare()"

Or via train.py:
    python train.py --action prepare_object365

Images are organized in patches. The conversion supports both
automatic download and pre-downloaded local files.

Official URLs may require a Chinese mainland connection. Objects365
annotation JSON files can also be obtained from Biendata/OpenDataLab
after registration (https://data.baai.ac.cn/details/Objects365_2020).
"""

import json
import os
import subprocess
import tarfile
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from .utils import logger

ANN_URLS = {
    "train": "https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/zhiyuan_objv2_train.tar.gz",
    "val": "https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/zhiyuan_objv2_val.json",
}

PATCH_URL_TMPL = {
    "train": "https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/patch{i}.tar.gz",
    "val_v1": "https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/images/v1/patch{i}.tar.gz",
    "val_v2": "https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/images/v2/patch{i}.tar.gz",
}

NUM_TRAIN_PATCHES = 51
NUM_VAL_V1 = 16
NUM_VAL_V2 = 27

ANN_FILES = {
    "train": "zhiyuan_objv2_train.json",
    "val": "zhiyuan_objv2_val.json",
}


def _try_download(url: str, dest: str, desc: str = ""):
    """Download a file using wget or curl. Returns True on success."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if os.path.exists(dest):
        return True
    logger.info(f"Downloading {desc or os.path.basename(dest)} ...")
    if subprocess.run(
        ["wget", "-q", "--show-progress", "-O", dest, url],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0:
        return True
    if subprocess.run(
        ["curl", "-fSL", "-o", dest, url],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0:
        return True
    if os.path.exists(dest):
        os.remove(dest)
    return False


def download_annotations(output_dir: str, splits: Tuple[str, ...] = ("train", "val")) -> str:
    """Download COCO-format annotation JSON files."""
    ann_dir = os.path.join(output_dir, "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    for split in splits:
        url = ANN_URLS.get(split)
        if not url:
            continue
        json_name = ANN_FILES[split]
        json_dest = os.path.join(ann_dir, json_name)

        # Check if valid JSON already exists (skip download)
        if os.path.exists(json_dest) and os.path.getsize(json_dest) > 0:
            try:
                with open(json_dest) as f:
                    json.loads(f.read(1024))
                continue
            except (json.JSONDecodeError, Exception):
                logger.warning(f"  {json_name} exists but is invalid, re-downloading")
                os.remove(json_dest)

        if split == "train":
            # Train annotations come as a tarball
            tarball = os.path.join(output_dir, "zhiyuan_objv2_train.tar.gz")
            if not _try_download(url, tarball, desc="Objects365 train annotations (tarball)"):
                logger.warning(f"Failed to download train annotations from {url}")
                logger.info("See README for alternative download methods.")
                continue
            logger.info("  Extracting train annotations ...")
            with tarfile.open(tarball) as tar:
                tar.extractall(path=ann_dir)
            os.remove(tarball)
        else:
            # Val annotations are a raw JSON
            if not _try_download(url, json_dest, desc=f"Objects365 {split} annotations"):
                logger.warning(f"Failed to download {split} annotations from {url}")
                continue

    return ann_dir


def download_images(objects_root: str, split: str, max_patches: Optional[int] = None):
    """Download and extract Objects365 image patches."""
    image_dir = os.path.join(objects_root, "images", split)
    os.makedirs(image_dir, exist_ok=True)

    if split == "train":
        num = NUM_TRAIN_PATCHES
    else:
        num = NUM_VAL_V1 + NUM_VAL_V2

    count = max_patches if max_patches else num

    for i in range(count):
        if split == "train":
            url = PATCH_URL_TMPL["train"].format(i=i)
        elif i < NUM_VAL_V1:
            url = PATCH_URL_TMPL["val_v1"].format(i=i)
        else:
            url = PATCH_URL_TMPL["val_v2"].format(i=i - NUM_VAL_V1)

        patch_name = f"patch{i}"
        patch_dir = os.path.join(image_dir, patch_name)

        if os.path.isdir(patch_dir) and len(os.listdir(patch_dir)) > 0:
            logger.info(f"  {patch_name} already exists, skipping")
            continue

        tarball = os.path.join(objects_root, f"{split}_{patch_name}.tar.gz")
        try:
            if not _try_download(url, tarball, desc=f"Objects365 {split} {patch_name}"):
                logger.warning(f"  Failed to download {patch_name}")
                continue
            logger.info(f"  Extracting {patch_name} ...")
            os.makedirs(patch_dir, exist_ok=True)
            with tarfile.open(tarball) as tar:
                tar.extractall(path=patch_dir)
            os.remove(tarball)
        except Exception as e:
            logger.warning(f"  Failed to download/extract {patch_name}: {e}")


def _build_image_index(image_dir: str) -> Dict[str, str]:
    """Build a mapping from filename to full path (walk once)."""
    index = {}
    for root, _dirs, files in os.walk(image_dir):
        for f in files:
            index[f] = os.path.join(root, f)
    return index


def resolve_image_path(image_dir: str, file_name: str, index: Dict[str, str]) -> Optional[str]:
    """Resolve image path using pre-built index."""
    path = os.path.join(image_dir, file_name)
    if os.path.exists(path):
        return os.path.abspath(path)
    basename = os.path.basename(file_name)
    full = index.get(basename)
    if full:
        return full
    return None


def build_category_map(ann_data: Dict) -> Dict[int, str]:
    """Build mapping from category_id to category name."""
    return {cat["id"]: cat["name"] for cat in ann_data.get("categories", [])}


def convert_to_jsonl(
    ann_path: str,
    image_dir: str,
    output_path: str,
    split: str = "train",
    max_boxes_per_image: int = 50,
    min_box_area: int = 400,
    max_images: Optional[int] = None,
):
    """Convert Objects365 COCO annotations to EdgeLocate JSONL format.

    Each sample uses the generic "Locate all instances" prompt like COCO.
    Objects365 uses 365 categories so prompts can be long.
    """
    logger.info(f"Loading annotations from {ann_path}")
    with open(ann_path) as f:
        ann_data = json.load(f)

    cat_map = build_category_map(ann_data)

    img_to_anns = defaultdict(list)
    for ann in ann_data.get("annotations", []):
        img_to_anns[ann["image_id"]].append(ann)

    images = ann_data.get("images", [])
    if max_images:
        images = images[:max_images]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    written = 0
    skipped = 0

    logger.info("Building image index (single walk) ...")
    img_index = _build_image_index(image_dir)
    logger.info(f"Indexed {len(img_index)} images")

    with open(output_path, "w") as out_f:
        for img in tqdm(images, desc=f"Objects365 {split} -> JSONL"):
            image_id = img["id"]
            fname = img.get("file_name", "")
            if not fname:
                continue

            img_path = resolve_image_path(image_dir, fname, img_index)
            if img_path is None:
                skipped += 1
                continue

            anns = img_to_anns.get(image_id, [])
            img_w = img.get("width", 640)
            img_h = img.get("height", 480)

            boxes_with_cats = []
            seen_cats = set()
            for ann in anns:
                bbox = ann.get("bbox", [])
                if len(bbox) < 4:
                    continue
                x, y, w, h = bbox[:4]
                if w * h < min_box_area:
                    continue
                cat_id = ann.get("category_id")
                cat_name = cat_map.get(cat_id, "object")
                x1 = int(max(0, x) * 1000 / img_w)
                y1 = int(max(0, y) * 1000 / img_h)
                x2 = int(min(img_w, x + w) * 1000 / img_w)
                y2 = int(min(img_h, y + h) * 1000 / img_h)
                x1 = min(1000, max(0, x1))
                y1 = min(1000, max(0, y1))
                x2 = min(1000, max(0, x2))
                y2 = min(1000, max(0, y2))
                boxes_with_cats.append((cat_name, [x1, y1, x2, y2]))
                seen_cats.add(cat_name)

            if not boxes_with_cats:
                skipped += 1
                continue

            if len(boxes_with_cats) > max_boxes_per_image:
                boxes_with_cats = boxes_with_cats[:max_boxes_per_image]

            gpt_parts = []
            for cat_name, box in boxes_with_cats:
                x1, y1, x2, y2 = box
                gpt_parts.append(
                    f"<ref>{cat_name}</ref><box><{x1}><{y1}><{x2}><{y2}></box>"
                )
            gpt_value = "".join(gpt_parts)

            cat_list = "</c>".join(sorted(seen_cats))
            human_value = f"Locate all the instances that matches the following description: {cat_list}."

            rel_img_path = os.path.relpath(img_path, os.path.dirname(output_path))
            if rel_img_path.startswith(".."):
                rel_img_path = os.path.abspath(img_path)

            sample = {
                "image": rel_img_path,
                "conversations": [
                    {"from": "human", "value": human_value},
                    {"from": "gpt", "value": gpt_value},
                ],
            }

            out_f.write(json.dumps(sample) + "\n")
            written += 1

    logger.info(f"Wrote {written} samples to {output_path} (skipped {skipped})")
    return written


def prepare(
    objects_root: str = "./data/objects365",
    output_dir: str = "./data/objects365_detection",
    splits: Tuple[str, ...] = ("train", "val"),
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    max_boxes_per_image: int = 50,
    download: bool = True,
    download_images_flag: bool = True,
    max_patches: Optional[int] = None,
):
    """Download Objects365 and convert to JSONL.

    Args:
        objects_root: Where Objects365 images and annotations live/will be downloaded
        output_dir: Where to write JSONL files
        splits: Which splits to process
        max_train: Limit train images
        max_val: Limit val images
        max_boxes_per_image: Cap boxes per sample
        download: Auto-download annotations
        download_images_flag: Auto-download images (can be very large)
        max_patches: Limit patches downloaded per split (for testing)
    """
    os.makedirs(output_dir, exist_ok=True)

    if download:
        ann_dir = download_annotations(objects_root, splits=splits)
    else:
        ann_dir = os.path.join(objects_root, "annotations")

    if download_images_flag:
        for split in splits:
            logger.info(f"Downloading images for {split} ...")
            download_images(objects_root, split, max_patches=max_patches)

    max_images_per_split = {}
    if max_train is not None:
        max_images_per_split["train"] = max_train
    if max_val is not None:
        max_images_per_split["val"] = max_val

    for split in splits:
        ann_path = os.path.join(ann_dir, ANN_FILES.get(split, f"zhiyuan_objv2_{split}.json"))
        if not os.path.exists(ann_path):
            logger.warning(f"Annotation file not found: {ann_path}")
            continue

        image_dir = os.path.join(objects_root, "images", split)
        if not os.path.isdir(image_dir):
            logger.warning(f"Image directory not found: {image_dir}")
            continue

        out_path = os.path.join(output_dir, f"{split}.jsonl")
        convert_to_jsonl(
            ann_path=ann_path,
            image_dir=image_dir,
            output_path=out_path,
            split=split,
            max_boxes_per_image=max_boxes_per_image,
            max_images=max_images_per_split.get(split),
        )

    logger.info(f"Done. Files in {output_dir}/")
    for fname in sorted(os.listdir(output_dir)):
        if fname.endswith(".jsonl"):
            fpath = os.path.join(output_dir, fname)
            count = sum(1 for _ in open(fpath))
            logger.info(f"  {fname}: {count} samples ({os.path.getsize(fpath)/1e6:.1f} MB)")


def add_object365_parser(subparsers):
    """Add object365 subcommand to argparse."""
    parser = subparsers.add_parser("prepare_object365", help="Prepare Objects365 detection dataset")
    parser.add_argument("--objects_root", default="./data/objects365", help="Objects365 data directory")
    parser.add_argument("--output_dir", default="./data/objects365_detection", help="Output directory")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to process")
    parser.add_argument("--max_train", type=int, default=None, help="Limit train images")
    parser.add_argument("--max_val", type=int, default=None, help="Limit val images")
    parser.add_argument("--max_boxes", type=int, default=50, help="Max boxes per image")
    parser.add_argument("--no-download", action="store_true", help="Skip download")
    parser.add_argument("--no-download-images", action="store_true", help="Skip image download")
    parser.add_argument("--max-patches", type=int, default=None, help="Limit patches downloaded per split")
    return parser

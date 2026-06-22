"""Create synthetic sample dataset in ShareGPT JSONL format for detection training."""

import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from .utils import ensure_dir, set_seed, logger

SHAPE_LABELS = [
    "rectangle", "square", "circle", "triangle",
    "pentagon", "star", "diamond", "ellipse",
]

SHAPE_COLORS = {
    "rectangle": (255, 100, 100),
    "square": (100, 255, 100),
    "circle": (100, 100, 255),
    "triangle": (255, 255, 100),
    "pentagon": (255, 100, 255),
    "star": (100, 255, 255),
    "diamond": (200, 200, 100),
    "ellipse": (200, 100, 200),
}


def generate_shape_based_boxes(
    shape_type: str,
    img_size: int = 224,
    canvas=None,
) -> Tuple[List[List[float]], Image.Image]:
    """Generate a colored shape and return its bounding box and the image."""
    if canvas is None:
        canvas = Image.new("RGB", (img_size, img_size), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    shapes = {
        "rectangle": lambda: _draw_rectangle(draw, img_size),
        "square": lambda: _draw_square(draw, img_size),
        "circle": lambda: _draw_circle(draw, img_size),
        "triangle": lambda: _draw_triangle(draw, img_size),
        "pentagon": lambda: _draw_pentagon(draw, img_size),
        "star": lambda: _draw_star(draw, img_size),
        "diamond": lambda: _draw_diamond(draw, img_size),
        "ellipse": lambda: _draw_ellipse(draw, img_size),
    }

    if shape_type not in shapes:
        shape_type = random.choice(list(shapes.keys()))

    bbox = shapes[shape_type]()
    return bbox, canvas


def _draw_rectangle(draw, img_size):
    pad = img_size // 8
    x1 = random.randint(pad, img_size // 3)
    y1 = random.randint(pad, img_size // 3)
    x2 = random.randint(2 * img_size // 3, img_size - pad)
    y2 = random.randint(2 * img_size // 3, img_size - pad)
    draw.rectangle([x1, y1, x2, y2], fill=SHAPE_COLORS["rectangle"])
    return [[x1, y1, x2, y2]]


def _draw_square(draw, img_size):
    pad = img_size // 6
    side = random.randint(30, img_size // 3)
    x1 = random.randint(pad, img_size - side - pad)
    y1 = random.randint(pad, img_size - side - pad)
    x2 = x1 + side
    y2 = y1 + side
    draw.rectangle([x1, y1, x2, y2], fill=SHAPE_COLORS["square"])
    return [[x1, y1, x2, y2]]


def _draw_circle(draw, img_size):
    r = random.randint(20, img_size // 4)
    cx = random.randint(r, img_size - r)
    cy = random.randint(r, img_size - r)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=SHAPE_COLORS["circle"])
    return [[cx - r, cy - r, cx + r, cy + r]]


def _draw_triangle(draw, img_size):
    pad = img_size // 6
    cx, cy = random.randint(pad, img_size - pad), random.randint(pad, img_size - pad)
    r = random.randint(30, img_size // 4)
    pts = [(cx + r * math.cos(2 * math.pi * i / 3 - math.pi / 2),
            cy + r * math.sin(2 * math.pi * i / 3 - math.pi / 2)) for i in range(3)]
    draw.polygon(pts, fill=SHAPE_COLORS["triangle"])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [[min(xs), min(ys), max(xs), max(ys)]]


def _draw_pentagon(draw, img_size):
    pad = img_size // 6
    cx, cy = random.randint(pad, img_size - pad), random.randint(pad, img_size - pad)
    r = random.randint(30, img_size // 4)
    pts = [(cx + r * math.cos(2 * math.pi * i / 5 - math.pi / 2),
            cy + r * math.sin(2 * math.pi * i / 5 - math.pi / 2)) for i in range(5)]
    draw.polygon(pts, fill=SHAPE_COLORS["pentagon"])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [[min(xs), min(ys), max(xs), max(ys)]]


def _draw_star(draw, img_size):
    pad = img_size // 6
    cx, cy = random.randint(pad, img_size - pad), random.randint(pad, img_size - pad)
    outer_r = random.randint(30, img_size // 4)
    inner_r = outer_r // 2
    pts = []
    for i in range(5):
        pts.append((cx + outer_r * math.cos(2 * math.pi * i / 5 - math.pi / 2),
                    cy + outer_r * math.sin(2 * math.pi * i / 5 - math.pi / 2)))
        pts.append((cx + inner_r * math.cos(2 * math.pi * (i + 0.5) / 5 - math.pi / 2),
                    cy + inner_r * math.sin(2 * math.pi * (i + 0.5) / 5 - math.pi / 2)))
    draw.polygon(pts, fill=SHAPE_COLORS["star"])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [[min(xs), min(ys), max(xs), max(ys)]]


def _draw_diamond(draw, img_size):
    pad = img_size // 6
    cx = random.randint(pad, img_size - pad)
    cy = random.randint(pad, img_size - pad)
    dx = random.randint(20, img_size // 4)
    dy = random.randint(20, img_size // 4)
    pts = [(cx, cy - dy), (cx + dx, cy), (cx, cy + dy), (cx - dx, cy)]
    draw.polygon(pts, fill=SHAPE_COLORS["diamond"])
    x1, y1, x2, y2 = cx - dx, cy - dy, cx + dx, cy + dy
    return [[x1, y1, x2, y2]]


def _draw_ellipse(draw, img_size):
    pad = img_size // 6
    x1 = random.randint(pad, img_size // 3)
    y1 = random.randint(pad, img_size // 3)
    x2 = random.randint(2 * img_size // 3, img_size - pad)
    y2 = random.randint(2 * img_size // 3, img_size - pad)
    draw.ellipse([x1, y1, x2, y2], fill=SHAPE_COLORS["ellipse"])
    return [[x1, y1, x2, y2]]


def create_sample_dataset(
    output_path: str,
    num_samples: int = 100,
    max_boxes_per_image: int = 8,
    img_size: int = 224,
    seed: int = 42,
):
    """Create a synthetic dataset with shape-based boxes.

    Generates images with colored shapes and their bounding boxes
    in <ref>label</ref><box><d><d><d><d></box> format.

    Each sample uses human/gpt roles with categories separated by </c>.
    """
    set_seed(seed)
    out_dir = os.path.dirname(output_path) or "."
    ensure_dir(out_dir)

    image_subdir = "syn_images"
    image_dir = os.path.join(out_dir, image_subdir)
    ensure_dir(image_dir)

    with open(output_path, "w") as f:
        for i in range(num_samples):
            img = Image.new("RGB", (img_size, img_size), color=(0, 0, 0))

            num_shapes = random.randint(1, min(3, max_boxes_per_image))
            boxes_with_labels = []

            for _ in range(num_shapes):
                label = random.choice(SHAPE_LABELS)
                bbox, img = generate_shape_based_boxes(label, img_size, canvas=img)
                for box in bbox:
                    boxes_with_labels.append((label, box))

            # Save image
            fname = f"sample_{i:04d}.png"
            abs_img_path = os.path.join(image_dir, fname)
            img.save(abs_img_path)

            # Build GPT response: <ref>label</ref><box><d><d><d><d></box> per box
            gpt_parts = []
            seen_labels = set()
            for label, box in boxes_with_labels:
                x1, y1, x2, y2 = [int(round(c * 1000 / img_size)) for c in box]
                x1 = max(0, min(1000, x1))
                y1 = max(0, min(1000, y1))
                x2 = max(0, min(1000, x2))
                y2 = max(0, min(1000, y2))
                gpt_parts.append(
                    f"<ref>{label}</ref><box><{x1}><{y1}><{x2}><{y2}></box>"
                )
                seen_labels.add(label)
            gpt_value = "".join(gpt_parts)

            # Build human prompt with unique categories
            cat_list = "</c>".join(sorted(seen_labels))
            human_value = f"Locate all the instances that matches the following description: {cat_list}."

            # Store relative image path
            rel_img_path = f"{image_subdir}/{fname}"

            sample = {
                "image": rel_img_path,
                "conversations": [
                    {"from": "human", "value": human_value},
                    {"from": "gpt", "value": gpt_value},
                ],
            }
            f.write(json.dumps(sample) + "\n")

    logger.info(f"Created {num_samples} synthetic samples at {output_path}")
    logger.info(f"Images saved to {image_dir}")
    return output_path

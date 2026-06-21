"""Create synthetic sample dataset in ShareGPT JSONL format for detection training."""

import json
import math
import os
import random
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw

from .utils import ensure_dir, set_seed, logger, boxes_to_tokens, SPECIAL_TOKENS

SAMPLE_CAPTIONS = [
    "Find all pedestrians in the scene.",
    "Detect vehicles in this image.",
    "Locate the traffic lights.",
    "Find all animals in the image.",
    "Detect all persons in the photo.",
    "Locate the stop signs.",
    "Find chairs and tables.",
    "Detect all bottles and cups.",
    "Locate all books on the shelf.",
    "Find all cars in the parking lot.",
    "Detect all cats in the picture.",
    "Locate all dogs in the image.",
    "Find all bicycles in the scene.",
    "Detect all umbrellas.",
    "Locate all street signs.",
]


def generate_random_boxes(
    num_boxes: int,
    img_size: int = 224,
    min_size: int = 10,
    max_size: int = 112,
) -> List[List[float]]:
    """Generate random non-overlapping boxes in [0, img_size] range."""
    boxes = []
    attempts = 0
    while len(boxes) < num_boxes and attempts < 100:
        x1 = random.uniform(0, img_size - min_size)
        y1 = random.uniform(0, img_size - min_size)
        w = random.uniform(min_size, min(max_size, img_size - x1))
        h = random.uniform(min_size, min(max_size, img_size - y1))
        x2 = x1 + w
        y2 = y1 + h

        overlap = False
        for bx1, by1, bx2, by2 in boxes:
            inter_x1 = max(x1, bx1)
            inter_y1 = max(y1, by1)
            inter_x2 = min(x2, bx2)
            inter_y2 = min(y2, by2)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                overlap = True
                break

        if not overlap:
            boxes.append([x1, y1, x2, y2])
        attempts += 1

    return boxes


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
    color = tuple(random.randint(100, 255) for _ in range(3))
    draw.rectangle([x1, y1, x2, y2], fill=color)
    return [[x1, y1, x2, y2]]


def _draw_square(draw, img_size):
    pad = img_size // 6
    side = random.randint(30, img_size // 3)
    x1 = random.randint(pad, img_size - side - pad)
    y1 = random.randint(pad, img_size - side - pad)
    x2 = x1 + side
    y2 = y1 + side
    color = tuple(random.randint(100, 255) for _ in range(3))
    draw.rectangle([x1, y1, x2, y2], fill=color)
    return [[x1, y1, x2, y2]]


def _draw_circle(draw, img_size):
    r = random.randint(20, img_size // 4)
    cx = random.randint(r, img_size - r)
    cy = random.randint(r, img_size - r)
    color = tuple(random.randint(100, 255) for _ in range(3))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    return [[cx - r, cy - r, cx + r, cy + r]]


def _draw_triangle(draw, img_size):
    pad = img_size // 6
    cx, cy = random.randint(pad, img_size - pad), random.randint(pad, img_size - pad)
    r = random.randint(30, img_size // 4)
    pts = [(cx + r * math.cos(2 * math.pi * i / 3 - math.pi / 2),
            cy + r * math.sin(2 * math.pi * i / 3 - math.pi / 2)) for i in range(3)]
    color = tuple(random.randint(100, 255) for _ in range(3))
    draw.polygon(pts, fill=color)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [[min(xs), min(ys), max(xs), max(ys)]]


def _draw_pentagon(draw, img_size):
    pad = img_size // 6
    cx, cy = random.randint(pad, img_size - pad), random.randint(pad, img_size - pad)
    r = random.randint(30, img_size // 4)
    pts = [(cx + r * math.cos(2 * math.pi * i / 5 - math.pi / 2),
            cy + r * math.sin(2 * math.pi * i / 5 - math.pi / 2)) for i in range(5)]
    color = tuple(random.randint(100, 255) for _ in range(3))
    draw.polygon(pts, fill=color)
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
    color = tuple(random.randint(100, 255) for _ in range(3))
    draw.polygon(pts, fill=color)
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
    color = tuple(random.randint(100, 255) for _ in range(3))
    draw.polygon(pts, fill=color)
    x1, y1, x2, y2 = cx - dx, cy - dy, cx + dx, cy + dy
    return [[x1, y1, x2, y2]]


def _draw_ellipse(draw, img_size):
    pad = img_size // 6
    x1 = random.randint(pad, img_size // 3)
    y1 = random.randint(pad, img_size // 3)
    x2 = random.randint(2 * img_size // 3, img_size - pad)
    y2 = random.randint(2 * img_size // 3, img_size - pad)
    color = tuple(random.randint(100, 255) for _ in range(3))
    draw.ellipse([x1, y1, x2, y2], fill=color)
    return [[x1, y1, x2, y2]]


def create_sample_dataset(
    output_path: str,
    num_samples: int = 100,
    max_boxes_per_image: int = 8,
    img_size: int = 224,
    seed: int = 42,
):
    """Create a synthetic dataset with shape-based boxes.

    Generates images with colored shapes (rectangle, circle, triangle, etc.)
    and their bounding boxes in discrete coordinate token format.
    """
    set_seed(seed)
    out_dir = os.path.dirname(output_path) or "."
    ensure_dir(out_dir)

    image_dir = os.path.join(out_dir, "syn_images")
    ensure_dir(image_dir)

    shape_types = ["rectangle", "square", "circle", "triangle", "pentagon", "star", "diamond", "ellipse"]

    from PIL import Image

    with open(output_path, "w") as f:
        for i in range(num_samples):
            img = Image.new("RGB", (img_size, img_size), color=(0, 0, 0))

            num_shapes = random.randint(1, min(3, max_boxes_per_image))
            all_boxes = []

            for _ in range(num_shapes):
                shape_type = random.choice(shape_types)
                bbox, img = generate_shape_based_boxes(shape_type, img_size, canvas=img)
                all_boxes.extend(bbox)

            img_path = os.path.join(image_dir, f"sample_{i:04d}.png")
            img.save(img_path)

            box_tokens = boxes_to_tokens(all_boxes, img_size)
            caption = random.choice(SAMPLE_CAPTIONS)

            # Build assistant response with discrete coordinate tokens
            assistant_text = box_tokens

            sample = {
                "image": img_path,
                "conversations": [
                    {"from": "user", "value": f"{SPECIAL_TOKENS['image']}\n{caption}"},
                    {
                        "from": "assistant",
                        "value": assistant_text,
                        "boxes": all_boxes,
                    },
                ],
            }
            f.write(json.dumps(sample) + "\n")

    logger.info(f"Created {num_samples} synthetic samples at {output_path}")
    logger.info(f"Images saved to {image_dir}")
    return output_path

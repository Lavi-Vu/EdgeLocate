import json
import logging
import os
import random
import re
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {
    "image": "<|image|>",
    "box_start": "<box>",
    "box_end": "</box>",
}

COORD_TOKENS = [f"<{i}>" for i in range(1001)]

LOCANY_SPECIAL_TOKENS = list(SPECIAL_TOKENS.values()) + COORD_TOKENS

COORD_START_ID = 151668
BOX_START_TOKEN_ID = 151666
BOX_END_TOKEN_ID = 151667
IMAGE_TOKEN_ID = 151665


def coord_to_token(coord: int) -> str:
    return f"<{coord}>"


def coord_to_token_id(coord: int) -> int:
    return COORD_START_ID + coord


def token_id_to_coord(token_id: int) -> int:
    return token_id - COORD_START_ID


def boxes_to_tokens(boxes: List[List[float]], img_size: int = 224) -> str:
    parts = []
    for box in boxes:
        x1 = int(round(box[0] * 1000 / img_size))
        y1 = int(round(box[1] * 1000 / img_size))
        x2 = int(round(box[2] * 1000 / img_size))
        y2 = int(round(box[3] * 1000 / img_size))
        x1 = max(0, min(1000, x1))
        y1 = max(0, min(1000, y1))
        x2 = max(0, min(1000, x2))
        y2 = max(0, min(1000, y2))
        parts.append(f"<box>{coord_to_token(x1)}{coord_to_token(y1)}{coord_to_token(x2)}{coord_to_token(y2)}</box>")
    return "".join(parts)


def parse_boxes_from_text(text: str) -> List[List[float]]:
    boxes = []
    seen = set()

    def add(b):
        t = tuple(b)
        if t not in seen:
            seen.add(t)
            boxes.append(b)

    # Pattern 1: <ref>label</ref><box><d><d><d><d></box>
    for m in re.finditer(r"<ref>[^<]*</ref><box><(\d+)><(\d+)><(\d+)><(\d+)></box>", text):
        add([int(g) for g in m.groups()])
    # Pattern 2: <box><d><d><d><d></box> (standalone boxes, no ref)
    for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", text):
        add([int(g) for g in m.groups()])
    # Pattern 3: <ref>label<d+><d+><d+><d+>... (malformed, missing </ref><box>)
    for m in re.finditer(r"<ref>[^<]*((?:<\d+>)+)(?:</box>|$)", text):
        coords = [int(x.group(1)) for x in re.finditer(r"<(\d+)>", m.group(1))]
        for i in range(0, len(coords) - len(coords) % 4, 4):
            add(coords[i:i+4])
    # Pattern 4: bare <d><d><d><d></box> (no ref at all)
    for m in re.finditer(r"<(\d+)><(\d+)><(\d+)><(\d+)></box>", text):
        add([int(g) for g in m.groups()])

    return boxes


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_model_size(model: torch.nn.Module, trainable_only: bool = False) -> str:
    if trainable_only:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        params = sum(p.numel() for p in model.parameters())
    if params >= 1e9:
        return f"{params / 1e9:.2f}B"
    return f"{params / 1e6:.2f}M"


@contextmanager
def timer(name: str = ""):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info(f"[{name}] took {elapsed:.3f}s")


def load_image(image_path: str, size: Tuple[int, int] = (224, 224)) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    return img


def denormalize_boxes(
    boxes: torch.Tensor, orig_sizes: torch.Tensor
) -> torch.Tensor:
    w = orig_sizes[:, 0:1]
    h = orig_sizes[:, 1:2]
    denorm = boxes.clone()
    denorm[..., 0::2] = denorm[..., 0::2] * w / 1000.0
    denorm[..., 1::2] = denorm[..., 1::2] * h / 1000.0
    return denorm

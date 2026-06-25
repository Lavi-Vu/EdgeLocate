import json
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, WeightedRandomSampler, ConcatDataset
from PIL import Image

from .config import DataConfig
from .utils import SPECIAL_TOKENS, COORD_TOKENS, load_image, logger


ROLE_MAP = {
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "gpt": "assistant",
}


def parse_sharegpt_line(
    line: Dict,
    image_dir: str,
    tokenizer,
    max_length: int = 2048,
    image_size: Tuple[int, int] = (224, 224),
    data_augment: bool = False,
) -> Optional[Dict]:
    """Parse a JSON line into model inputs with discrete coordinate tokens.

    Supports:
      Format 1 (conversations): {image, conversations: [{from, value}]}
        Roles: user/human, assistant/gpt
        Boxes inline via <ref>cat</ref><box><d><d><d><d></box> in GPT text.
      Format 2 (messages): {messages: [{role, content}]}
    """
    conversations = line.get("conversations", [])
    messages = line.get("messages", [])

    if not conversations and not messages:
        return None

    image = None

    if messages:
        user_text = ""
        assistant_text = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                user_text = content
            elif role == "assistant":
                assistant_text = content

        if not assistant_text or not user_text:
            return None

        from .utils import parse_boxes_from_text
        if not parse_boxes_from_text(assistant_text):
            return None

        image_path = line.get("image_path", "")
        if not image_path:
            for msg in messages:
                if msg.get("role") == "user" and "image_path" in msg:
                    image_path = msg["image_path"]
                    break
    else:
        image_path = line.get("image", "")
        if not image_path or not conversations:
            return None

        user_text = ""
        assistant_text = ""
        for conv in conversations:
            role = ROLE_MAP.get(conv.get("from", ""), conv.get("from", ""))
            if role == "user":
                user_text = conv["value"]
            elif role == "assistant":
                assistant_text = conv["value"]

        if not assistant_text:
            return None

        # Parse boxes from inline <ref>cat</ref><box><d><d><d><d></box> format
        from .utils import parse_boxes_from_text
        if not parse_boxes_from_text(assistant_text):
            return None

    # Resolve image path
    resolved = image_path
    if not os.path.isabs(resolved):
        resolved = os.path.join(image_dir, resolved)
    if not os.path.exists(resolved):
        basename = os.path.basename(image_path)
        resolved = os.path.join(image_dir, basename)
    if not os.path.exists(resolved):
        return None

    try:
        use_size = image_size
        # Data augmentation: random scale crop for robustness
        if data_augment:
            import random as _r
            scale = _r.uniform(0.8, 1.0)
            ih = int(image_size[0] * scale)
            iw = int(image_size[1] * scale)
            image = load_image(resolved, size=(ih, iw))
            from torchvision import transforms as T
            image = T.Compose([
                T.Resize(image_size),
                T.RandomCrop(image_size),
            ])(image)
        else:
            image = load_image(resolved, size=image_size)
    except Exception as e:
        logger.warning(f"Failed to load image {resolved}: {e}")
        return None

    if image is None:
        return None

    # Prepend image token to user text if not already present
    image_token = SPECIAL_TOKENS["image"]
    if image_token not in user_text:
        user_text = f"{image_token}\n{user_text}"

    if hasattr(tokenizer, "apply_chat_template"):
        chat_messages = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
        try:
            full_text = tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            full_text = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n{assistant_text}<|im_end|>"
    else:
        full_text = f"User: {user_text}\nAssistant: {assistant_text}"

    enc = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", [1] * len(input_ids))

    # Find assistant response start for label masking
    assistant_token_id = tokenizer.convert_tokens_to_ids("assistant")
    assistant_start = -1
    for i in range(len(input_ids)):
        if input_ids[i] == assistant_token_id:
            assistant_start = i + 1
            break

    if assistant_start < 0:
        assistant_start = len(input_ids) // 2

    labels = [-100] * len(input_ids)
    labels[assistant_start:] = input_ids[assistant_start:]

    return {
        "pixel_values": image,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class _SubDataset(Dataset):
    """Internal wrapper holding one dataset's lines and config."""

    def __init__(self, data_lines: List[Dict], image_dir: str, tokenizer, max_length: int,
                 image_size: Tuple[int, int], data_augment: bool):
        self.data = data_lines
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size
        self.data_augment = data_augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        for attempt in range(5):
            line = self.data[idx]
            parsed = parse_sharegpt_line(
                line, self.image_dir, self.tokenizer,
                self.max_length, self.image_size,
                data_augment=self.data_augment,
            )
            if parsed is not None:
                break
            idx = (idx + 1) % len(self.data)
        else:
            raise RuntimeError("Could not find a valid sample after 5 attempts")

        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        pixel_values = transform(parsed["pixel_values"])

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(parsed["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(parsed["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(parsed["labels"], dtype=torch.long),
        }


class DetectionDataset(Dataset):
    """Multi-dataset detection training dataset supporting data recipes.

    Supports:
      - Single JSONL via data_path/image_dir
      - Data recipe JSON via data_recipe (dict or path) with per-dataset:
          annotation: str (path to JSONL)
          root: str (image directory)
          repeat_time: float (relative sampling weight)
          data_augment: bool (random scale crop)
    """

    def __init__(
        self,
        data_path: str = "",
        image_dir: str = "",
        tokenizer=None,
        max_length: int = 2048,
        image_size: Tuple[int, int] = (224, 224),
        data_recipe: Optional[Dict] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size

        datasets = []

        # Load from recipe
        if data_recipe:
            for name, cfg in data_recipe.items():
                ann = cfg.get("annotation", "")
                root = cfg.get("root", "")
                repeat = float(cfg.get("repeat_time", 1.0))
                augment = bool(cfg.get("data_augment", False))
                if not ann or not os.path.exists(ann):
                    logger.warning(f"Recipe '{name}': annotation not found at {ann}, skipping")
                    continue
                lines = []
                with open(ann) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            lines.append(json.loads(line))
                logger.info(f"Recipe '{name}': {len(lines)} samples, root={root}, "
                            f"repeat={repeat}, augment={augment}")
                sub = _SubDataset(lines, root, tokenizer, max_length, image_size, augment)
                if repeat > 1:
                    datasets.append(ConcatDataset([sub] * int(repeat)))
                elif repeat > 0:
                    datasets.append(sub)

        # Load single JSONL (legacy path)
        if data_path and os.path.exists(data_path):
            lines = []
            with open(data_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lines.append(json.loads(line))
            logger.info(f"Loaded {len(lines)} samples from {data_path} (legacy path)")
            sub = _SubDataset(lines, image_dir, tokenizer, max_length, image_size, False)
            datasets.append(sub)
        elif data_path and not data_recipe:
            logger.warning(f"Data path not found: {data_path}")

        if not datasets:
            logger.warning("No data loaded!")
            self.dataset = None
        elif len(datasets) == 1:
            self.dataset = datasets[0]
        else:
            self.dataset = ConcatDataset(datasets)

    def __len__(self) -> int:
        return len(self.dataset) if self.dataset is not None else 0

    def __getitem__(self, idx: int) -> Dict:
        return self.dataset[idx]

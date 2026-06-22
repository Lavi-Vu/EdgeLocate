import json
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
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


class DetectionDataset(Dataset):
    """Dataset for detection training from JSONL file."""

    def __init__(
        self,
        data_path: str,
        image_dir: str,
        tokenizer,
        max_length: int = 2048,
        image_size: Tuple[int, int] = (224, 224),
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size
        self.image_dir = image_dir

        self.data = []
        if data_path:
            with open(data_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.data.append(json.loads(line))
            logger.info(f"Loaded {len(self.data)} samples from {data_path}")
        else:
            logger.warning("No data path provided")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        for attempt in range(5):
            line = self.data[idx]
            parsed = parse_sharegpt_line(
                line, self.image_dir, self.tokenizer,
                self.max_length, self.image_size,
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

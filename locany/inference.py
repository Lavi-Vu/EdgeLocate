import re
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw
from transformers import GenerationConfig

from .config import InferenceConfig, ModelConfig
from .model import LocateAnythingForDetection
from .utils import SPECIAL_TOKENS, load_image, logger


class DetectionInferenceEngine:
    """Inference engine for detection model using standard autoregressive generation."""

    def __init__(
        self,
        model: LocateAnythingForDetection,
        tokenizer,
        config: InferenceConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device

        self.model.set_image_token_id(tokenizer)

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        text: str,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Dict:
        """Run detection inference on an image.

        Args:
            image: PIL Image
            text: Text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)

        Returns:
            Dict with "text" (generated response) and "boxes" (list of [x1,y1,x2,y2])
        """
        orig_w, orig_h = image.size
        ve_size = self.model.vision_encoder.image_size

        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((ve_size, ve_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        pixel_values = transform(image).unsqueeze(0).to(self.device)

        if SPECIAL_TOKENS["image"] not in text:
            if "<image>" in text:
                text = text.replace("<image>", SPECIAL_TOKENS["image"])
            else:
                text = f"{SPECIAL_TOKENS['image']}\n{text}"

        messages = [{"role": "user", "content": text}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = formatted["input_ids"].to(self.device)
        attention_mask = torch.ones_like(input_ids)

        max_new = max_new_tokens or self.config.max_new_tokens

        gen_config = GenerationConfig(
            max_new_tokens=max_new,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        outputs = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config,
        )

        full_ids = outputs.sequences if hasattr(outputs, "sequences") else outputs
        text_output = self.tokenizer.decode(full_ids[0], skip_special_tokens=False)

        boxes = self._parse_boxes(text_output, orig_w, orig_h)

        return {
            "text": text_output,
            "boxes": boxes,
            "sequences": full_ids,
        }

    def _parse_boxes(self, text: str, img_w: int, img_h: int) -> List[List[float]]:
        """Parse boxes from generated text and scale to original image size."""
        from .utils import parse_boxes_from_text
        boxes = parse_boxes_from_text(text)
        return [[
            int(b[0] * img_w / 1000),
            int(b[1] * img_h / 1000),
            int(b[2] * img_w / 1000),
            int(b[3] * img_h / 1000),
        ] for b in boxes]

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Image.Image],
        texts: List[str],
        batch_size: int = 8,
    ) -> List[Dict]:
        """Run batched detection inference.

        All images in a call share the same prompt text (texts[0] is used).
        Processes in mini-batches of batch_size.

        Args:
            images: List of PIL Images
            texts: List of text prompts (only first is used, shared across batch)
            batch_size: Max images per generate call

        Returns:
            List of Dicts with "text" and "boxes"
        """
        from torchvision import transforms
        ve_size = self.model.vision_encoder.image_size
        transform = transforms.Compose([
            transforms.Resize((ve_size, ve_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        text = texts[0] if texts else ""
        if SPECIAL_TOKENS["image"] not in text:
            if "<image>" in text:
                text = text.replace("<image>", SPECIAL_TOKENS["image"])
            else:
                text = f"{SPECIAL_TOKENS['image']}\n{text}"

        messages = [{"role": "user", "content": text}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        )
        prompt_ids = formatted["input_ids"].to(self.device)
        prompt_mask = torch.ones_like(prompt_ids)

        gen_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        results = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]
            batch_orig_sizes = [img.size for img in batch_imgs]

            pixel_values = torch.stack([transform(img) for img in batch_imgs]).to(self.device)
            batch_ids = prompt_ids.expand(len(batch_imgs), -1).contiguous()
            batch_mask = prompt_mask.expand(len(batch_imgs), -1).contiguous()

            outputs = self.model.generate(
                pixel_values=pixel_values,
                input_ids=batch_ids,
                attention_mask=batch_mask,
                generation_config=gen_config,
            )

            full_ids = outputs.sequences if hasattr(outputs, "sequences") else outputs
            for j, seq in enumerate(full_ids):
                text_out = self.tokenizer.decode(seq, skip_special_tokens=False)
                orig_w, orig_h = batch_orig_sizes[j]
                boxes = self._parse_boxes(text_out, orig_w, orig_h)
                results.append({"text": text_out, "boxes": boxes})

        return results


def visualize_boxes(
    image: Image.Image,
    boxes: List[List[float]],
    output_path: Optional[str] = None,
) -> Image.Image:
    """Draw boxes on image. Boxes are in original image pixel coordinates."""
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    if output_path:
        image.save(output_path)
    return image


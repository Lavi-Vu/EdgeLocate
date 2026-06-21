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
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
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

        boxes = self._parse_boxes(text_output)

        return {
            "text": text_output,
            "boxes": boxes,
            "sequences": full_ids,
        }

    def _parse_boxes(self, text: str) -> List[List[float]]:
        """Parse boxes from generated text: <box><d><d><d><d></box>."""
        from .utils import parse_boxes_from_text
        return parse_boxes_from_text(text)

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Image.Image],
        texts: List[str],
    ) -> List[Dict]:
        results = []
        for img, txt in zip(images, texts):
            results.append(self.predict(img, txt))
        return results


def visualize_boxes(
    image: Image.Image,
    boxes: List[List[float]],
    output_path: Optional[str] = None,
) -> Image.Image:
    """Draw boxes on image. Boxes are in [0, 1000] normalized coords."""
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for box in boxes:
        x1, y1, x2, y2 = box
        x1_px = int(x1 * w / 1000)
        y1_px = int(y1 * h / 1000)
        x2_px = int(x2 * w / 1000)
        y2_px = int(y2 * h / 1000)
        draw.rectangle([x1_px, y1_px, x2_px, y2_px], outline="red", width=3)
    if output_path:
        image.save(output_path)
    return image

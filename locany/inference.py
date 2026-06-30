import re
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import GenerationConfig

from .config import InferenceConfig, ModelConfig
from .model import LocateAnythingForDetection
from .utils import SPECIAL_TOKENS, load_image, logger
from .generate_utils import get_token_ids_from_config


class DetectionInferenceEngine:
    def __init__(self, model: LocateAnythingForDetection, tokenizer, config: InferenceConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
        self.model.set_image_token_id(tokenizer)

    @torch.no_grad()
    def predict(self, image: Image.Image, text: str, max_new_tokens: Optional[int] = None,
                temperature: float = 0.0, top_p: float = 1.0) -> Dict:
        orig_w, orig_h = image.size
        ve = self.model.vision_encoder
        is_moonvit = self.model.is_moonvit

        if is_moonvit:
            pixel_values, grid_hws = self._preprocess_moonvit(image, ve)
            pixel_values = pixel_values.unsqueeze(0).to(self.device)
            grid_hws = grid_hws.to(self.device)
        else:
            from torchvision import transforms
            ve_size = ve.image_size
            transform = transforms.Compose([
                transforms.Resize((ve_size, ve_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            pixel_values = transform(image).unsqueeze(0).to(self.device)
            grid_hws = None

        if SPECIAL_TOKENS["image"] not in text:
            if "<image>" in text:
                text = text.replace("<image>", SPECIAL_TOKENS["image"])
            else:
                text = f"{SPECIAL_TOKENS['image']}\n{text}"

        messages = [{"role": "user", "content": text}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        )
        input_ids = formatted["input_ids"].to(self.device)
        attention_mask = torch.ones_like(input_ids)

        max_new = max_new_tokens or self.config.max_new_tokens

        generation_mode = self.config.mode
        if generation_mode != 'slow' and is_moonvit:
            # Use PBD generation for MoonViT
            generated_ids = self.model.generate_pbd(
                pixel_values=pixel_values, input_ids=input_ids,
                attention_mask=attention_mask, tokenizer=self.tokenizer,
                generation_mode=generation_mode, max_new_tokens=max_new,
                temperature=temperature or self.config.temperature,
                top_p=top_p or self.config.top_p,
                block_size=self.model.model_config.block_size,
            )
            text_output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        else:
            gen_config = GenerationConfig(
                max_new_tokens=max_new, do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            outputs = self.model.generate(
                pixel_values=pixel_values, input_ids=input_ids,
                attention_mask=attention_mask, generation_config=gen_config,
            )
            full_ids = outputs.sequences if hasattr(outputs, "sequences") else outputs
            text_output = self.tokenizer.decode(full_ids[0], skip_special_tokens=False)

        boxes = self._parse_boxes(text_output, orig_w, orig_h)
        return {"text": text_output, "boxes": boxes}

    def _preprocess_moonvit(self, image: Image.Image, ve) -> Tuple[torch.Tensor, torch.Tensor]:
        from .modeling_vit import MoonVitPretrainedModel
        ps = ve._patch_size
        kh, kw = ve.merge_kernel_size
        w, h = image.size
        pad_h = (kh * ps - h % (kh * ps)) % (kh * ps)
        pad_w = (kw * ps - w % (kw * ps)) % (kw * ps)
        if pad_h > 0 or pad_w > 0:
            from torchvision.transforms.functional import pad
            image = pad(image, (0, 0, pad_w, pad_h), fill=0)
            w, h = image.size
        image = image.resize((w, h), Image.LANCZOS)
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        pixel_values = transform(image)
        grid_h = h // ps
        grid_w = w // ps
        grid_hws = torch.tensor([[grid_h, grid_w]], dtype=torch.long)
        return pixel_values, grid_hws

    def _parse_boxes(self, text: str, img_w: int, img_h: int) -> List[List[float]]:
        from .utils import parse_boxes_from_text
        boxes = parse_boxes_from_text(text)
        return [[
            int(b[0] * img_w / 1000), int(b[1] * img_h / 1000),
            int(b[2] * img_w / 1000), int(b[3] * img_h / 1000),
        ] for b in boxes]

    @torch.no_grad()
    def predict_batch(self, images: List[Image.Image], texts: List[str], batch_size: int = 8) -> List[Dict]:
        from torchvision import transforms
        ve = self.model.vision_encoder
        is_moonvit = self.model.is_moonvit
        ve_size = ve.image_size

        if not is_moonvit:
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
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        )
        prompt_ids = formatted["input_ids"].to(self.device)
        prompt_mask = torch.ones_like(prompt_ids)

        gen_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens, do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        results = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]
            batch_orig_sizes = [img.size for img in batch_imgs]

            if is_moonvit:
                pixel_list = []
                grid_list = []
                for img in batch_imgs:
                    pv, gh = self._preprocess_moonvit(img, ve)
                    pixel_list.append(pv)
                    grid_list.append(gh)
                pixel_values = torch.stack(pixel_list).to(self.device)
            else:
                pixel_values = torch.stack([transform(img) for img in batch_imgs]).to(self.device)

            batch_ids = prompt_ids.expand(len(batch_imgs), -1).contiguous()
            batch_mask = prompt_mask.expand(len(batch_imgs), -1).contiguous()

            outputs = self.model.generate(
                pixel_values=pixel_values, input_ids=batch_ids,
                attention_mask=batch_mask, generation_config=gen_config,
            )
            full_ids = outputs.sequences if hasattr(outputs, "sequences") else outputs
            for j, seq in enumerate(full_ids):
                text_out = self.tokenizer.decode(seq, skip_special_tokens=False)
                orig_w, orig_h = batch_orig_sizes[j]
                boxes = self._parse_boxes(text_out, orig_w, orig_h)
                results.append({"text": text_out, "boxes": boxes})

        return results


def visualize_boxes(image: Image.Image, boxes: List[List[float]],
                    labels: Optional[List[str]] = None,
                    output_path: Optional[str] = None) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        if labels and i < len(labels) and labels[i]:
            label = labels[i]
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.rectangle([x1, y1 - th - 4, x1 + tw + 4, y1], fill="red")
            draw.text((x1 + 2, y1 - th - 2), label, fill="white", font=font)
    if output_path:
        image.save(output_path)
    return image

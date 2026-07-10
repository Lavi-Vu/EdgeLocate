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
        token_ids_config = get_token_ids_from_config(self.model.model_config)
        box_start_id = token_ids_config['box_start_token_id']
        box_end_id = token_ids_config['box_end_token_id']
        coord_start_id = token_ids_config['coord_start_token_id']
        coord_end_id = token_ids_config['coord_end_token_id']

        if generation_mode != 'slow':
            generated_ids, confs = self.model.generate_pbd(
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

        # Extract per-box confidences from generated token IDs
        if generation_mode != 'slow' and confs:
            box_confs = self._extract_box_confidences(
                generated_ids[0], confs, box_start_id, box_end_id,
                coord_start_id, coord_end_id,
            )
        else:
            box_confs = [0.0] * len(boxes)

        return {"text": text_output, "boxes": boxes, "confidences": box_confs}

    def _extract_box_confidences(self, token_ids, confs, box_start_id, box_end_id,
                                  coord_start_id, coord_end_id):
        """Walk generated token IDs to find boxes and compute per-box confidence."""
        box_confs = []
        i = 0
        n = len(token_ids)
        coord_range = range(coord_start_id, coord_end_id + 1)
        while i < n:
            if token_ids[i] == box_start_id and i + 5 < n:
                if all(token_ids[i + j + 1] in coord_range for j in range(4)) and token_ids[i + 5] == box_end_id:
                    coord_confs = [confs[i + j + 1] for j in range(4)]
                    box_confs.append(sum(coord_confs) / len(coord_confs))
                    i += 6
                    continue
            i += 1
        return box_confs

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
            int(round(b[0] * img_w / 1000)), int(round(b[1] * img_h / 1000)),
            int(round(b[2] * img_w / 1000)), int(round(b[3] * img_h / 1000)),
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

        prompt_data = []
        for text in texts:
            if SPECIAL_TOKENS["image"] not in text:
                if "<image>" in text:
                    text = text.replace("<image>", SPECIAL_TOKENS["image"])
                else:
                    text = f"{SPECIAL_TOKENS['image']}\n{text}"
            messages = [{"role": "user", "content": text}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
            )
            prompt_data.append(formatted["input_ids"].to(self.device))

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

            batch_ids = prompt_data[i:i + batch_size]
            max_len = max(p.shape[1] for p in batch_ids)
            padded_ids = torch.full((len(batch_ids), max_len), self.tokenizer.pad_token_id,
                                    dtype=torch.long, device=self.device)
            padded_mask = torch.zeros((len(batch_ids), max_len), device=self.device)
            for k, pids in enumerate(batch_ids):
                slen = pids.shape[1]
                padded_ids[k, :slen] = pids[0]
                padded_mask[k, :slen] = 1

            outputs = self.model.generate(
                pixel_values=pixel_values, input_ids=padded_ids,
                attention_mask=padded_mask, generation_config=gen_config,
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
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
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


def visualize_prediction(image: Image.Image, pred_boxes: List[List[float]],
                         gt_boxes: List[List[float]],
                         pred_labels: Optional[List[str]] = None,
                         gt_labels: Optional[List[str]] = None,
                         output_path: Optional[str] = None) -> Image.Image:
    """Draw predicted (red) and ground truth (green) boxes on the same image."""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, box in enumerate(gt_boxes):
        x1, y1, x2, y2 = map(int, box)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
        if gt_labels and i < len(gt_labels) and gt_labels[i]:
            label = f"GT: {gt_labels[i]}"
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.rectangle([x1, y1 - th - 2, x1 + tw + 2, y1], fill="lime")
            draw.text((x1 + 1, y1 - th - 1), label, fill="black", font=font)

    for i, box in enumerate(pred_boxes):
        x1, y1, x2, y2 = map(int, box)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        if pred_labels and i < len(pred_labels) and pred_labels[i]:
            label = f"PRED: {pred_labels[i]}"
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.rectangle([x1, y1 - th - 2, x1 + tw + 2, y1], fill="red")
            draw.text((x1 + 1, y1 - th - 1), label, fill="white", font=font)

    if output_path:
        image.save(output_path)
    return image

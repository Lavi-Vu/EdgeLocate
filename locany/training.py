import os
import random
from typing import Optional

import torch
from transformers import Trainer, TrainingArguments, TrainerCallback
from torch.nn.utils.rnn import pad_sequence

import json
from PIL import Image, ImageDraw, ImageFont
from .config import ModelConfig, TrainingConfig
from .model import LocateAnythingForDetection
from .utils import logger, set_seed, load_image


IGNORE_INDEX = -100


class TrainVisCallback:
    """Saves original vs augmented training images at epoch end."""

    def __init__(self, dataset, image_dir, save_dir, num_samples=8):
        self.dataset = dataset
        self.image_dir = image_dir
        self.save_dir = save_dir
        self.num_samples = num_samples
        os.makedirs(save_dir, exist_ok=True)
        self._font = None

    def _resolve(self, path):
        resolved = path if os.path.isabs(path) else os.path.join(self.image_dir, path)
        if not os.path.exists(resolved):
            resolved = os.path.join(self.image_dir, os.path.basename(path))
        return resolved if os.path.exists(resolved) else None

    @property
    def font(self):
        if self._font is None:
            try:
                self._font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            except (OSError, IOError):
                self._font = ImageFont.load_default()
        return self._font

    def on_epoch_end(self, epoch):
        raw_lines = self.dataset.data if hasattr(self.dataset, 'data') else []
        if not raw_lines:
            return
        n_imgs = min(4, len(raw_lines))
        chosen = random.sample(raw_lines, n_imgs)

        def _maybe_augment(img):
            w, h = img.size
            if random.random() < 0.5:
                target = random.randint(640, 2560)
                long_edge = max(w, h)
                if long_edge != target:
                    s = target / long_edge
                    img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
            return img.resize((224, 224), Image.LANCZOS)

        rows = []
        for sample in chosen:
            resolved = self._resolve(sample.get("image", ""))
            if not resolved:
                continue
            orig = Image.open(resolved).convert("RGB")
            ow, oh = orig.size

            tiles = []
            # Tile 0: no-augment baseline
            base = Image.new("RGB", (224, 224), "white")
            base.paste(orig.resize((224, 224), Image.LANCZOS), (0, 0))
            draw = ImageDraw.Draw(base)
            draw.text((2, 2), f"no aug", fill="red", font=self.font)
            draw.text((2, 14), f"{ow}x{oh}", fill="red", font=self.font)
            tiles.append(base)

            # Tiles 1-8: 8 independent augmentations
            for _ in range(8):
                aug = _maybe_augment(orig.copy())
                tile = Image.new("RGB", (224, 224), "white")
                tile.paste(aug, (0, 0))
                tiles.append(tile)

            rows.append(tiles)

        if not rows:
            return
        grid_w = 9 * 228
        grid_h = len(rows) * 228
        grid = Image.new("RGB", (grid_w, grid_h), "gray")
        for ri, r in enumerate(rows):
            for ci, t in enumerate(r):
                grid.paste(t, (ci * 228 + 2, ri * 228 + 2))
        out = os.path.join(self.save_dir, f"epoch_{epoch}.jpg")
        grid.save(out)
        logger.info(f"Saved training pipeline vis: {out}")


class _DetectionTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super()._save(output_dir, state_dict)
        if output_dir is None:
            output_dir = self.args.output_dir
        save_adapter_and_extra(self.model, output_dir, self.processing_class)


def save_adapter_and_extra(model, output_dir: str, tokenizer=None):
    if hasattr(model, "llm") and hasattr(model.llm, "peft_config"):
        model.llm.save_pretrained(output_dir)
        logger.info(f"LoRA adapter saved to {output_dir}")
        non_llm_state = {k: v for k, v in model.state_dict().items() if not k.startswith("llm.")}
        extra_state = {}
        for k, v in model.state_dict().items():
            if k.startswith("llm.") and "lora_" not in k and ("lm_head" in k or "embed_tokens" in k):
                extra_state[k] = v
        save_dict = {**non_llm_state, **extra_state}
        if save_dict:
            torch.save(save_dict, os.path.join(output_dir, "non_llm.pt"))
            logger.info(f"Non-LoRA weights saved ({len(save_dict)} keys)")
    else:
        model.save_pretrained(output_dir)
        logger.info(f"Full model saved to {output_dir}")
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)


def setup_training(model, model_cfg: ModelConfig, train_cfg: TrainingConfig,
                   train_dataset, eval_dataset=None, data_collator=None, tokenizer=None):
    set_seed(train_cfg.seed)

    if train_cfg.gradient_checkpointing:
        if hasattr(model.llm, "gradient_checkpointing_enable"):
            model.llm.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled on LLM")
        if model.is_moonvit:
            model.vision_encoder.encoder.gradient_checkpointing = True
            logger.info("Gradient checkpointing enabled on MoonViT")

    deepspeed_config = train_cfg.deepspeed
    if deepspeed_config and not os.path.exists(deepspeed_config):
        logger.warning(f"DeepSpeed config not found: {deepspeed_config}")
        deepspeed_config = None

    if tokenizer is not None:
        processing_class = tokenizer
    else:
        processing_class = None

    training_args = TrainingArguments(
        output_dir=train_cfg.output_dir,
        num_train_epochs=train_cfg.num_epochs,
        per_device_train_batch_size=train_cfg.per_device_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        warmup_ratio=train_cfg.warmup_ratio,
        weight_decay=train_cfg.weight_decay,
        bf16=train_cfg.bf16,
        fp16=train_cfg.fp16 if not train_cfg.bf16 else False,
        gradient_checkpointing=False,
        deepspeed=deepspeed_config,
        logging_steps=train_cfg.logging_steps,
        save_steps=train_cfg.save_steps,
        eval_steps=train_cfg.eval_steps,
        save_total_limit=train_cfg.save_total_limit,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        max_grad_norm=train_cfg.max_grad_norm,
        remove_unused_columns=False,
        report_to=["tensorboard"] if train_cfg.logging_steps > 0 else [],
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
        dataloader_pin_memory=False,
    )

    if data_collator is None:
        data_collator = DetectionDataCollator(tokenizer)

    trainer = _DetectionTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=data_collator, processing_class=processing_class,
    )

    # Add training visualization callback (start + each epoch)
    vis_dir = os.path.join(train_cfg.output_dir, "epoch_vis")
    _vis_helper = TrainVisCallback(train_dataset, train_dataset.image_dir, vis_dir)

    class _VisCB(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            _vis_helper.on_epoch_end(0)
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            _vis_helper.on_epoch_end(int(state.epoch))
            return control

    trainer.add_callback(_VisCB())

    return trainer


class DetectionDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = {}
        batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])
        input_ids = [f["input_ids"] if isinstance(f["input_ids"], torch.Tensor) else torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [f["attention_mask"] if isinstance(f["attention_mask"], torch.Tensor) else torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = [f["labels"] if isinstance(f["labels"], torch.Tensor) else torch.tensor(f["labels"], dtype=torch.long) for f in features]
        batch["input_ids"] = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch["attention_mask"] = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        batch["labels"] = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return batch


class PackedDataCollator:
    """Collator for PackedDetectionDataset that stacks sub_sample_lengths and position_ids."""

    def __call__(self, features):
        batch = {}
        batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])
        batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
        batch["labels"] = torch.stack([f["labels"] for f in features])
        batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
        batch["position_ids"] = torch.stack([f["position_ids"] for f in features])
        batch["sub_sample_lengths"] = torch.stack([f["sub_sample_lengths"] for f in features])
        return batch


def save_model(trainer: Trainer, output_dir: str, tokenizer=None, model_cfg: Optional[ModelConfig] = None):
    os.makedirs(output_dir, exist_ok=True)
    save_adapter_and_extra(trainer.model, output_dir, tokenizer)
    if model_cfg is not None:
        cfg_path = os.path.join(output_dir, "locany_config.json")
        with open(cfg_path, "w") as f:
            json.dump(model_cfg.to_dict(), f, indent=2)
        logger.info(f"Config saved to {cfg_path}")

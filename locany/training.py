import os
from typing import Optional

import torch
from transformers import Trainer, TrainingArguments
from torch.nn.utils.rnn import pad_sequence

import json
from .config import ModelConfig, TrainingConfig
from .model import LocateAnythingForDetection
from .utils import logger, set_seed


class _DetectionTrainer(Trainer):
    """Custom Trainer that saves LoRA adapter and non-llm weights with every checkpoint."""

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super()._save(output_dir, state_dict)
        if output_dir is None:
            output_dir = self.args.output_dir
        save_adapter_and_extra(self.model, output_dir, self.processing_class)


def save_adapter_and_extra(model, output_dir: str, tokenizer=None):
    """Save LoRA adapter (if present), non-LLM weights, tokenizer, and config."""
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


def setup_training(
    model: LocateAnythingForDetection,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    train_dataset,
    eval_dataset=None,
    data_collator=None,
    tokenizer=None,
) -> Trainer:
    """Set up the HF Trainer for detection training."""
    set_seed(train_cfg.seed)

    if train_cfg.gradient_checkpointing:
        if hasattr(model.llm, "gradient_checkpointing_enable"):
            model.llm.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled on LLM")

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
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processing_class,
    )

    return trainer


class DetectionDataCollator:
    """Data collator for detection training. Pads sequences, stacks images."""

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
        batch["labels"] = pad_sequence(labels, batch_first=True, padding_value=-100)

        return batch


def save_model(trainer: Trainer, output_dir: str, tokenizer=None, model_cfg: Optional[ModelConfig] = None):
    os.makedirs(output_dir, exist_ok=True)
    save_adapter_and_extra(trainer.model, output_dir, tokenizer)
    if model_cfg is not None:
        cfg_path = os.path.join(output_dir, "locany_config.json")
        with open(cfg_path, "w") as f:
            json.dump(model_cfg.to_dict(), f, indent=2)
        logger.info(f"Config saved to {cfg_path}")

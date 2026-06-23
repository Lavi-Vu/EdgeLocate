#!/usr/bin/env python3
"""
LocateAnything - Training & Inference with discrete coordinate tokens.

Uses the standard VLM approach (VE → Projector → LLM+LoRA → LM head)
with box coordinates predicted as discrete tokens <0>–<1000> via cross-entropy loss.
"""

import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from locany import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    InferenceConfig,
    parse_args,
    create_model,
    load_model_from_dir,
    DetectionDataset,
    setup_training,
    save_model,
    DetectionInferenceEngine,
    evaluate_model,
    create_sample_dataset,
    prepare_refcoco,
    prepare_coco,
    set_seed,
    get_model_size,
    load_image,
    LOCANY_SPECIAL_TOKENS,
    COORD_TOKENS,
)

for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger("train_locany")


def setup_tokenizer(model_cfg: ModelConfig):
    """Load tokenizer and add coordinate tokens plus special tokens."""
    from transformers import AutoTokenizer
    from locany.utils import SPECIAL_TOKENS, COORD_TOKENS, LOCANY_SPECIAL_TOKENS

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.llm_model,
        trust_remote_code=True,
        padding_side="right",
        use_fast=True,
    )

    special_tokens_dict = {"additional_special_tokens": LOCANY_SPECIAL_TOKENS}
    tokenizer.add_special_tokens(special_tokens_dict)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Tokenizer loaded: {model_cfg.llm_model}")
    logger.info(f"Vocab size: {len(tokenizer)}")
    logger.info(f"Added {len(COORD_TOKENS)} coordinate tokens + special tokens")

    return tokenizer


def resize_model_embeddings(model, tokenizer):
    """Resize model embeddings to accommodate new tokens."""
    old_vocab = model.llm.get_input_embeddings().weight.shape[0]
    new_vocab = len(tokenizer)
    if new_vocab > old_vocab:
        model.llm.resize_token_embeddings(new_vocab)
        logger.info(f"Resized embeddings: {old_vocab} -> {new_vocab}")

        with torch.no_grad():
            new_embeddings = model.llm.get_input_embeddings()
            for i in range(old_vocab, new_vocab):
                new_embeddings.weight[i].normal_(mean=0.0, std=0.02)
        logger.info("Initialized new token embeddings")


def run_training(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    data_cfg: DataConfig,
):
    """Run detection model training."""
    set_seed(train_cfg.seed)
    logger.info("=" * 60)
    logger.info("Starting Detection Training (discrete coordinate tokens)")
    logger.info(f"LLM: {model_cfg.llm_model}")
    logger.info(f"VE: {model_cfg.ve_model}")
    logger.info(f"Output: {train_cfg.output_dir}")

    tokenizer = setup_tokenizer(model_cfg)

    model = create_model(model_cfg)

    resize_model_embeddings(model, tokenizer)

    # Set image token ID on model
    model.set_image_token_id(tokenizer)

    total = get_model_size(model)
    trainable = get_model_size(model, trainable_only=True)
    logger.info(f"Model size: {total} total, {trainable} trainable")

    train_dataset = DetectionDataset(
        data_path=data_cfg.train_data_path,
        image_dir=data_cfg.image_dir,
        tokenizer=tokenizer,
        max_length=data_cfg.max_length,
    )

    eval_dataset = None
    if data_cfg.eval_data_path:
        eval_dataset = DetectionDataset(
            data_path=data_cfg.eval_data_path,
            image_dir=data_cfg.image_dir,
            tokenizer=tokenizer,
            max_length=data_cfg.max_length,
        )

    trainer = setup_training(
        model=model,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    save_model(trainer, train_cfg.output_dir, tokenizer=tokenizer, model_cfg=model_cfg)

    logger.info("Training complete!")
    return model, tokenizer


def load_model_for_inference(model_cfg: ModelConfig, output_dir: str, tokenizer):
    """Load model from saved weights."""
    if output_dir and os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "adapter_config.json")):
        model = load_model_from_dir(output_dir, tokenizer)
        logger.info(f"Loaded model from {output_dir}")
    elif output_dir and os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "model.safetensors")):
        model = load_model_from_dir(output_dir, tokenizer)
        logger.info(f"Loaded model from {output_dir}")
    else:
        model = create_model(model_cfg)
        resize_model_embeddings(model, tokenizer)
        model.set_image_token_id(tokenizer)
        logger.warning("No saved model weights found, using randomly initialized model")
    return model


def run_inference(
    model_cfg: ModelConfig,
    infer_cfg: InferenceConfig,
    data_cfg: DataConfig,
    output_dir: str = "",
):
    """Run inference on a sample."""
    logger.info("=" * 60)
    logger.info("Starting Detection Inference")

    tokenizer = setup_tokenizer(model_cfg)

    model = load_model_for_inference(model_cfg, output_dir, tokenizer)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Using device: {device}")

    engine = DetectionInferenceEngine(model, tokenizer, infer_cfg)

    if data_cfg.image_dir:
        import glob
        images = glob.glob(os.path.join(data_cfg.image_dir, "*.*"))
        if images:
            img_path = images[0]
            logger.info(f"Running inference on: {img_path}")
            from PIL import Image
            image = Image.open(img_path).convert("RGB")
            text = "Locate all the instances that matches the following description: all objects."

            result = engine.predict(image, text)
            logger.info(f"Generated text: {result['text'][:200]}...")
            logger.info(f"Detected {len(result['boxes'])} boxes")
            for i, box in enumerate(result["boxes"]):
                logger.info(f"  Box {i}: [{', '.join(f'{c:.1f}' for c in box)}]")
        else:
            logger.warning("No images found in image_dir")
    else:
        logger.warning("No image_dir provided for inference")


def run_evaluation(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    data_cfg: DataConfig,
):
    """Run model evaluation."""
    logger.info("=" * 60)
    logger.info("Starting Evaluation")

    tokenizer = setup_tokenizer(model_cfg)
    model = create_model(model_cfg)
    model.eval()
    resize_model_embeddings(model, tokenizer)
    model.set_image_token_id(tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    eval_dataset = DetectionDataset(
        data_path=data_cfg.eval_data_path or data_cfg.train_data_path,
        image_dir=data_cfg.image_dir,
        tokenizer=tokenizer,
        max_length=data_cfg.max_length,
    )

    results = evaluate_model(model, tokenizer, eval_dataset, max_samples=50)
    logger.info("Evaluation Results:")
    for k, v in results.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


def main():
    model_cfg, train_cfg, data_cfg, infer_cfg, action, no_download = parse_args()

    import sys
    extra_args = {}
    if "--num_samples" in sys.argv:
        idx = sys.argv.index("--num_samples")
        extra_args["num_samples"] = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 100
    if "--max_boxes_per_image" in sys.argv:
        idx = sys.argv.index("--max_boxes_per_image")
        extra_args["max_boxes_per_image"] = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 8

    if action == "train":
        run_training(model_cfg, train_cfg, data_cfg)
    elif action == "inference":
        run_inference(model_cfg, infer_cfg, data_cfg, output_dir=train_cfg.output_dir)
    elif action == "eval":
        run_evaluation(model_cfg, train_cfg, data_cfg)
    elif action == "create_sample":
        create_sample_dataset(
            output_path=data_cfg.train_data_path or "./sample_data.jsonl",
            num_samples=extra_args.get("num_samples", 100),
            max_boxes_per_image=extra_args.get("max_boxes_per_image", 8),
        )
    elif action == "prepare_refcoco":
        prepare_refcoco(
            coco_root=data_cfg.image_dir or "./data/coco",
            output_dir=train_cfg.output_dir,
            download=not no_download,
        )
    elif action == "prepare_coco":
        prepare_coco(
            coco_root=data_cfg.image_dir or "./data/coco",
            output_dir=train_cfg.output_dir,
            download=not no_download,
        )
    else:
        logger.error(f"Unknown action: {action}")


if __name__ == "__main__":
    main()

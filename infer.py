#!/usr/bin/env python3
"""Inference pipeline: load model, run detection on image with prompt."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from locany import (
    ModelConfig,
    InferenceConfig,
    load_model_from_dir,
    DetectionInferenceEngine,
    visualize_boxes,
    load_image,
    LOCANY_SPECIAL_TOKENS,
    SPECIAL_TOKENS,
)
from transformers import AutoTokenizer


def setup_tokenizer(model_cfg: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.llm_model,
        trust_remote_code=True,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.add_special_tokens({"additional_special_tokens": LOCANY_SPECIAL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="LocateAnything inference")
    parser.add_argument("--model_dir", default="./outputs_coord", help="Path to saved model directory")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prompt", default="Detect all objects in this image.", help="Text prompt")
    parser.add_argument("--output", default=None, help="Path to save visualized output")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {args.model_dir} ...")
    tokenizer = setup_tokenizer(ModelConfig())

    if os.path.exists(os.path.join(args.model_dir, "adapter_config.json")):
        tokenizer = setup_tokenizer(ModelConfig())
        model = load_model_from_dir(args.model_dir, tokenizer)
    elif os.path.exists(args.model_dir):
        tokenizer = setup_tokenizer(ModelConfig())
        model = load_model_from_dir(args.model_dir, tokenizer)
    else:
        print(f"Model directory not found: {args.model_dir}")
        sys.exit(1)

    model = model.to(device)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

    infer_cfg = InferenceConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    engine = DetectionInferenceEngine(model, tokenizer, infer_cfg)

    print(f"Loading image: {args.image}")
    image = load_image(args.image)
    print(f"Running inference with prompt: {args.prompt}")
    result = engine.predict(image, args.prompt)

    print(f"\nGenerated text: {result['text']}")
    print(f"Detected {len(result['boxes'])} boxes:")
    for i, box in enumerate(result["boxes"]):
        x1, y1, x2, y2 = box
        print(f"  [{i}] ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})")

    if args.output or not args.output:
        output_path = args.output or f"output_{os.path.splitext(os.path.basename(args.image))[0]}.png"
        vis = visualize_boxes(image.copy(), result["boxes"], output_path=output_path)
        print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    main()

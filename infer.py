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
    parse_labels_and_boxes,
    LOCANY_SPECIAL_TOKENS,
    SPECIAL_TOKENS,
)
from locany.utils import setup_tokenizer
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="LocateAnything inference")
    parser.add_argument("--model_dir", default="./outputs_coord", help="Path to saved model directory")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prompt", default="Detect all objects in this image.", help="Text prompt")
    parser.add_argument("--output", default=None, help="Path to save visualized output")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--mode", default="hybrid", choices=["fast", "hybrid", "slow"],
                        help="Generation mode: fast (PBD only), hybrid (PBD+AR fallback), slow (AR only)")
    parser.add_argument("--ve_model", default=None,
                        help="Override vision encoder model (e.g., google/siglip2-base-patch16-naflex)")
    parser.add_argument("--llm_model", default=None,
                        help="Override LLM model (e.g., Qwen/Qwen2.5-0.5B-Instruct)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Read saved config from checkpoint, with CLI overrides
    import json
    cfg_path = os.path.join(args.model_dir, "locany_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg_dict = json.load(f)
        model_cfg = ModelConfig.from_dict(cfg_dict)
        print(f"Loaded config: VE={model_cfg.ve_model}, LLM={model_cfg.llm_model}")
    else:
        model_cfg = ModelConfig()
        print(f"No locany_config.json found, using default: VE={model_cfg.ve_model}")
    if args.ve_model:
        model_cfg.ve_model = args.ve_model
        print(f"Overriding VE: {model_cfg.ve_model}")
    if args.llm_model:
        model_cfg.llm_model = args.llm_model
        print(f"Overriding LLM: {model_cfg.llm_model}")

    print(f"Loading model from {args.model_dir} ...")
    tokenizer = setup_tokenizer(model_cfg)
    model = load_model_from_dir(args.model_dir, tokenizer, model_cfg=model_cfg)

    model = model.to(device)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

    infer_cfg = InferenceConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        mode=args.mode,
    )

    engine = DetectionInferenceEngine(model, tokenizer, infer_cfg)

    print(f"Loading image: {args.image}")
    image = Image.open(args.image).convert("RGB")
    print(f"Running inference with prompt: {args.prompt}")
    result = engine.predict(image, args.prompt)

    print(f"\nGenerated text: {result['text']}")
    print(f"Detected {len(result['boxes'])} boxes:")
    confs = result.get('confidences', [])
    for i, box in enumerate(result["boxes"]):
        x1, y1, x2, y2 = box
        conf = confs[i] if i < len(confs) else 0.0
        print(f"  [{i}] ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})  conf={conf:.3f}")

    output_path = args.output or f"output_{os.path.splitext(os.path.basename(args.image))[0]}.png"
    label_boxes = parse_labels_and_boxes(result["text"])
    labels = [lb[0] for lb in label_boxes]
    vis = visualize_boxes(image.copy(), result["boxes"], labels=labels, output_path=output_path)
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    main()

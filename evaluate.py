#!/usr/bin/env python3
"""Benchmark a trained model on detection datasets.

Usage:
  # Benchmark on a JSONL dataset
  python evaluate.py \\
    --model_dir ./outputs \\
    --data ./data/coco_detection/val.jsonl \\
    --image_dir ./data/coco/val2017 \\
    --max_samples 200

  # Quick eval on 50 samples
  python evaluate.py \\
    --model_dir ./outputs \\
    --data ./data/coco_detection/val.jsonl \\
    --image_dir ./data/coco/val2017 \\
    --max_samples 50
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from locany import (
    ModelConfig,
    InferenceConfig,
    load_model_from_dir,
    DetectionInferenceEngine,
    run_benchmark,
    benchmark_on_jsonl,
    LOCANY_SPECIAL_TOKENS,
)
from transformers import AutoTokenizer


def setup_tokenizer(model_cfg: ModelConfig):
    from locany.utils import setup_tokenizer as _setup
    return _setup(model_cfg)


def main():
    parser = argparse.ArgumentParser(description="EdgeLocate Benchmark")
    parser.add_argument("--model_dir", default="./outputs", help="Path to saved model")
    parser.add_argument("--data", required=True, help="Path to JSONL dataset")
    parser.add_argument("--image_dir", default="", help="Image directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for faster eval")
    parser.add_argument("--output", default=None, help="Save results JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model from {args.model_dir} ...")
    tokenizer = setup_tokenizer(ModelConfig())
    model = load_model_from_dir(args.model_dir, tokenizer)
    model = model.to(device)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

    image_dir = args.image_dir or os.path.dirname(args.data)

    print(f"Benchmark on: {args.data}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print()

    results = benchmark_on_jsonl(
        model, tokenizer,
        jsonl_path=args.data,
        image_dir=image_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )

    print("\n=== Results ===")
    print(f"  Samples:      {results.get('num_samples', 0)}")
    print(f"  Mean IoU:     {results.get('mean_iou', 0):.4f}")
    print(f"  AP:           {results.get('AP', 0):.4f}")
    print(f"  Precision:    {results.get('Precision', 0):.4f}")
    print(f"  Recall:       {results.get('Recall', 0):.4f}")
    print(f"  F1:           {results.get('F1', 0):.4f}")
    print()
    for th in [0.5, 0.75, 0.9]:
        key_f1 = f"F1@{th:.2f}"
        key_ap = f"AP@{th:.2f}"
        if key_f1 in results:
            print(f"  @{th:.2f}  AP={results[key_ap]:.4f}  Precision={results.get(f'Precision@{th:.2f}', 0):.4f}  Recall={results.get(f'Recall@{th:.2f}', 0):.4f}  F1={results[key_f1]:.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

# EdgeLocate — Discrete Coordinate Token Detection

A <1B LocateAnything variant using **discrete coordinate tokens** (like EAGLE) instead of a regression head. Box coordinates are predicted as vocabulary tokens `<0>`–`<1000>` via standard cross-entropy loss through the LM head.

Default base: **Qwen2.5-0.5B-Instruct** + **SigLIP-Base-Patch16-224** + 2-layer MLP projector + LoRA.

## Architecture

```
                                ┌──────────────────────────────────────┐
  Image ──► Vision Encoder ──► MLP Projector ──┐                      │
                                │              ▼                       │
  Text ──► Tokenizer ──► Embedding ──► Qwen2.5 LLM+LoRA ──► LM Head ──► Token IDs
                                │              ▲                       │
                                └── <|image|> anchor tokens ───────────┘
```

| Component | Model | Params | Default Freeze |
|---|---|---|---|
| Vision Encoder | `google/siglip-base-patch16-224` (or SigLIP2, MobileCLIP, ...) | 92.88M | ✅ Frozen |
| MLP Projector | 2-layer Linear+GELU (VE hidden → LLM hidden) | ~1.5M | ❌ Trainable |
| LLM | `Qwen/Qwen2.5-0.5B-Instruct` | 494.03M | ❌ LoRA (r=64–128) |
| LM Head | Shared/untied output projection | 0.14M (untied) | ❌ Trainable |
| **Total** | | **~589M** | **~37M trainable** |

## How It Works

1. **Tokenization**: 1001 discrete coordinate tokens `<0>`–`<1000>` (IDs 151668–152668) plus `<box>` (151666), `</box>` (151667), `<ref>` (151669), and `</ref>` (151670) are added to the vocabulary
2. **Training**: Standard autoregressive next-token prediction. The GPT response contains `<ref>label</ref><box><d1><d2><d3><d4></box>` sequences where each `<d>` is a quantized coordinate in [0, 1000]
3. **Generation**: Standard `llm.generate()` with `inputs_embeds` (visual features replace the `<|image|>` token position). The model auto-regressively produces coordinate tokens.
4. **Box parsing**: Regex extracts `<ref>label</ref><box><(\d+)><(\d+)><(\d+)><(\d+)></box>` patterns from generated text, with fallbacks for malformed output.

## Setup

```bash
pip install torch transformers accelerate pillow torchvision peft safetensors tensorboard
```

## Quick Start

### Create synthetic data
```bash
python train.py --action create_sample --train_data_path ./data.jsonl --num_samples 200
```

### Train (default VE)
```bash
python train.py --action train \
  --train_data_path ./data.jsonl \
  --image_dir . \
  --output_dir ./outputs \
  --num_epochs 20 \
  --per_device_batch_size 1 \
  --learning_rate 5e-5 \
  --lora_r 64 --lora_alpha 128
```

### Train with a different Vision Encoder
```bash
python train.py --action train \
  --ve_model "google/siglip2-base-patch16-224" \
  --train_data_path ./data.jsonl \
  --image_dir . \
  --output_dir ./outputs \
  --num_epochs 20 \
  --per_device_batch_size 1 \
  --learning_rate 5e-5
```

Supported VEs:
- `google/siglip-base-patch16-224` (default)
- `google/siglip2-base-patch16-224`
- `google/siglip2-base-patch16-naflex`
- `google/siglip-so400m-patch14-384`
- `apple/MobileCLIP2-B`

### Inference (via `infer.py`)
```bash
python infer.py \
  --model_dir ./outputs \
  --image path/to/image.jpg \
  --prompt "Locate all the instances that matches the following description: all objects."
```

### COCO Detection
```bash
# Prepare train.jsonl and val.jsonl from COCO 2017
python train.py --action prepare_coco \
  --image_dir ./data/coco \
  --output_dir ./data/coco_detection \
  --max_train 50000 --max_val 1000

# Train on COCO detection (50k images)
python train.py --action train \
  --train_data_path ./data/coco_detection/train.jsonl \
  --image_dir ./data/coco/train2017 \
  --output_dir ./outputs_coco \
  --num_epochs 5 \
  --per_device_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 3e-5 \
  --lora_r 64
```

### Benchmark / Evaluation
```bash
# Evaluate trained model on COCO val set (batched, ~8x faster)
python evaluate.py \
  --model_dir ./outputs_coco \
  --data ./data/coco_detection/val.jsonl \
  --image_dir ./data/coco/val2017 \
  --batch_size 8 \
  --max_samples 500 \
  --output ./results.json
```

Outputs per-IoU-threshold metrics (AP@0.5, F1@0.75, etc.) plus mean AP, Precision, Recall, F1 across IoU 0.5:0.95.

**Metrics explained:**
- `AP` — mean Average Precision across IoU thresholds 0.50:0.05:0.95 (COCO primary)
- `F1` — mean F1 across the same thresholds
- `AP@0.50`, `F1@0.50` — metrics at IoU 0.5 (PASCAL VOC standard)
- `AP@0.75`, `F1@0.75` — stricter IoU for precise localization
- `AP@0.90`, `F1@0.90` — near-perfect localization

### RefCOCO+
```bash
# Prepare dataset (downloads COCO + refcoco+ annotations)
python train.py --action prepare_refcoco \
  --image_dir ./data/coco \
  --output_dir ./data/refcoco_plus

# Train on refcoco+ (referring expression comprehension)
python train.py --action train \
  --train_data_path ./data/refcoco_plus/train.jsonl \
  --image_dir ./data/coco \
  --output_dir ./outputs_refcoco \
  --num_epochs 10 \
  --per_device_batch_size 1 \
  --learning_rate 5e-5 \
  --lora_r 64
```

### Inference (via `train.py`)
```bash
python train.py --action inference \
  --image_dir ./images \
  --output_dir ./outputs
```

## Data Format

ShareGPT-style JSONL with `<ref>` and `<box>` tokens in the GPT response:

```json
{
  "image": "coco/train2017/000001.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "Locate all the instances that matches the following description: car</c>person</c>bicycle."
    },
    {
      "from": "gpt",
      "value": "<ref>car</ref><box><120><200><450><500></box><ref>person</ref><box><50><100><200><600></box>"
    }
  ]
}
```

- **Roles**: `human` / `gpt` (also accepts `user` / `assistant` for backward compatibility)
- **Prompt**: Categories listed separated by `</c>`
- **Response**: `<ref>label</ref><box><d1><d2><d3><d4></box>` per instance
- **Coordinates**: In [0, 1000] range, quantized to integer bins
- **Image path**: Relative to `image_dir` or absolute

## Key Design Decisions

- **Discrete tokens over regression**: Cross-entropy loss provides strong per-class supervision through the full model (LM head → LLM → projector → VE). Unlike MSE regression, gradients flow to every component.
- **LoRA on LLM** (r=64–128): The LLM must learn image-dependent hidden states at coordinate token positions. LoRA makes this feasible on a 4GB GPU.
- **Untied LM head**: `tie_word_embeddings=False` allows the LM head for new coordinate tokens to be trained independently from input embeddings.
- **Frozen VE**: Vision encoder stays frozen to save memory; projector and LoRA adapt visual features.
- **Standard autoregressive generation**: No PBD/MTP for now — simple `llm.generate()` with `inputs_embeds`.

## Programmatic API

```python
from locany import ModelConfig, load_model_from_dir, DetectionInferenceEngine, InferenceConfig, LOCANY_SPECIAL_TOKENS
from transformers import AutoTokenizer

# Load model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer.add_special_tokens({"additional_special_tokens": LOCANY_SPECIAL_TOKENS})
model = load_model_from_dir("./outputs", tokenizer).cuda().eval()

# Run inference
engine = DetectionInferenceEngine(model, tokenizer, InferenceConfig())
result = engine.predict(image, "Locate all the instances that matches the following description: all objects.")
print(result["boxes"])  # [[x1, y1, x2, y2], ...] in original image pixel coords

# Visualize
from locany import visualize_boxes
visualize_boxes(image, result["boxes"], output_path="output.png")
```

## All CLI Options

```
Model:
  --llm_model                     LLM backbone (default: Qwen/Qwen2.5-0.5B-Instruct)
  --ve_model                      Vision encoder (default: google/siglip-base-patch16-224)
  --freeze_llm / --no-freeze_llm  Freeze LLM backbone (default: False)
  --freeze_vision_encoder         Freeze VE (default: True)
  --use_lora / --no-lora          Use LoRA on LLM (default: True)
  --lora_r, --lora_alpha          LoRA rank and alpha (default: 128, 256)
  --attn_implementation           {sdpa,flash_attention_2,eager}
  --torch_dtype                   {float32,float16,bfloat16}

Vision Encoders:
  google/siglip-base-patch16-224      (default, 224px, 768 dim)
  google/siglip2-base-patch16-224      (SigLIP2, 224px)
  google/siglip2-base-patch16-naflex   (SigLIP2 + FlexiViT)
  google/siglip-so400m-patch14-384     (So400m, 384px, 1152 dim)
  apple/MobileCLIP2-B                  (MobileCLIP)

Training:
  --output_dir, --num_epochs, --per_device_batch_size
  --learning_rate, --warmup_ratio, --weight_decay
  --bf16 / --no-bf16, --gradient_checkpointing
  --logging_steps, --save_steps, --save_total_limit
  --lr_scheduler_type, --max_grad_norm

Data:
  --train_data_path, --eval_data_path, --image_dir
  --max_length

Inference:
  --max_new_tokens, --temperature, --top_p

Actions:
  --action {train,inference,eval,create_sample,prepare_refcoco,prepare_coco}
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers 4.45+ (for SigLIP2 support)
- accelerate, Pillow, torchvision
- peft (for LoRA)
- Optional: deepspeed, tensorboard

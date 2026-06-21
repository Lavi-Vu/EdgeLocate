# EdgeLocate — Discrete Coordinate Token Detection

A <1B LocateAnything variant using **discrete coordinate tokens** (like EAGLE) instead of a regression head. Box coordinates are predicted as vocabulary tokens `<0>`–`<1000>` via standard cross-entropy loss through the LM head.

Base model: **Qwen2.5-0.5B-Instruct** + **SigLIP-Base-Patch16-224** + 2-layer MLP projector + LoRA.

## Architecture

```
                                ┌──────────────────────────────────────┐
  Image ──► SigLIP Vision ──► MLP Projector ──┐                       │
                                │              ▼                       │
  Text ──► Tokenizer ──► Embedding ──► Qwen2.5 LLM+LoRA ──► LM Head ──► Token IDs
                                │              ▲                       │
                                └── <|image|> anchor tokens ───────────┘
```

| Component | Model | Params | Default Freeze |
|---|---|---|---|
| Vision Encoder | `google/siglip-base-patch16-224` | 92.88M | ✅ Frozen |
| MLP Projector | 2-layer Linear+GELU (768→896) | 1.49M | ❌ Trainable |
| LLM | `Qwen/Qwen2.5-0.5B-Instruct` | 494.03M | ❌ LoRA (r=64–128) |
| LM Head | Shared/untied output projection | 0.14M (untied) | ❌ Trainable |
| **Total** | | **589.62M** | **~36.7M trainable** |

## How It Works

1. **Tokenization**: 1001 discrete coordinate tokens `<0>`–`<1000>` (IDs 151668–152668) plus `<box>` (151666) and `</box>` (151667) are added to the vocabulary
2. **Training**: Standard autoregressive next-token prediction. The assistant response contains `<box><d1><d2><d3><d4></box>` sequences where each `<d>` is a quantized coordinate in [0, 1000]
3. **Generation**: Standard `llm.generate()` with `inputs_embeds` (visual features replace the `<|image|>` token position). The model auto-regressively produces coordinate tokens.
4. **Box parsing**: Regex extracts `<box><(\d+)><(\d+)><(\d+)><(\d+)></box>` patterns from generated text.

## Setup

```bash
pip install torch transformers accelerate pillow torchvision peft safetensors tensorboard
```

## Quick Start

### Create synthetic data
```bash
python train.py --action create_sample --train_data_path ./data.jsonl --num_samples 200
```

### Train
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

### Inference (via `infer.py`)
```bash
python infer.py \
  --model_dir ./outputs \
  --image path/to/image.jpg \
  --prompt "Detect all objects."
```

### COCO Detection
```bash
# Prepare dataset (downloads COCO 2017 train+val)
python train.py --action prepare_coco \
  --image_dir ./data/coco \
  --output_dir ./data/coco_detection \
  --max_train 50000 --max_val 1000

# Train on COCO detection (50k images, multiple objects per image)
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

ShareGPT-style JSONL with `<box>` tokens in the assistant response:

```json
{
  "image": "path/to/image.jpg",
  "conversations": [
    {
      "from": "user",
      "value": "<|image|>\nDetect all cats in this image."
    },
    {
      "from": "assistant",
      "value": "<box><120><340><560><780></box><box><890><450><100><768></box>",
      "boxes": [[120, 340, 560, 780], [890, 450, 100, 768]]
    }
  ]
}
```

Coordinates are in [0, 1000] range, quantized to integer bins.

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
result = engine.predict(image, "Detect all objects.")
print(result["boxes"])  # [[x1, y1, x2, y2], ...] in [0, 1000]

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
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers 4.38+
- accelerate, Pillow, torchvision
- peft (for LoRA)
- Optional: deepspeed, tensorboard

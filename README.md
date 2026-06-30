# EdgeLocate — Discrete Coordinate Token Detection

A <1B LocateAnything variant using **discrete coordinate tokens** (like EAGLE) instead of a regression head. Box coordinates are predicted as vocabulary tokens `<0>`–`<1000>` via standard cross-entropy loss through the LM head.

Dual vision encoder support: **SigLIP** (fixed 224×224, legacy) or **MoonViT** (native-resolution, 1152-dim, 27-layer) with optional **Parallel Box Decoding (PBD/MTP)** generation.

## Architecture

```
                                   ┌──────────────────────────────────────┐
  Image ──► SigLIP/MoonViT VE ──► MLP Projector ──┐                      │
                                   │              ▼                       │
  Text ──► Tokenizer ──► Embedding ──► Qwen2.5 LLM+LoRA ──► LM Head ──► Token IDs
                                   │              ▲                       │
                                   └── <|image|> anchor tokens ───────────┘

  Optional: PBD/MTP mode replaces single token sampling with parallel
  block decoding (block_size=6) using non-causal multi-token prediction
  masks for faster box generation.
```

| Component | Model | Params (SigLIP) | Params (MoonViT) | Default Freeze |
|---|---|---|---|---|
| Vision Encoder | SigLIP-Base-P16-224 or MoonViT-SO-400M | 92.88M | 408.15M | ✅ Frozen |
| VE LoRA | Optional backbone LoRA (r=8–16) | 0 | ~6M | Optional |
| MLP Projector | 2–4 layer Linear+GELU (VE hidden → LLM hidden) | ~1.5M | ~1.8M | ❌ Trainable |
| LLM | `Qwen/Qwen2.5-0.5B-Instruct` + LoRA (r=128) | 494.03M | 494.03M | ❌ LoRA |
| LM Head | Untied output projection | 0.14M | 0.14M | ❌ Trainable |
| **Total** | | **~589M** | **~904M** | **~37–44M trainable** |

## How It Works

1. **Tokenization**: 1001 discrete coordinate tokens `<0>`–`<1000>` (IDs 151670–152670) plus `<box>` (151666), `</box>` (151667), `<ref>` (151668), and `</ref>` (151669) are added to the vocabulary
2. **Training**: Standard autoregressive next-token prediction. The GPT response contains `<ref>label</ref><box><d1><d2><d3><d4></box>` sequences where each `<d>` is a quantized coordinate in [0, 1000]
3. **Generation (AR)**: Standard `llm.generate()` with `inputs_embeds` (visual features replace the `<|image|>` token position). The model auto-regressively produces coordinate tokens.
4. **Generation (PBD/MTP)**: Parallel Box Decoding predicts up to `block_size` tokens at once using multi-token prediction masks. Supports `fast` (pure MTP), `hybrid` (MTP with AR fallback), and `slow` (pure AR) modes.
5. **Box parsing**: Regex extracts `<ref>label</ref><box><(\d+)><(\d+)><(\d+)><(\d+)></box>` patterns from generated text, with fallbacks for malformed output.

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
- `google/siglip-base-patch16-224` (default, 224px, 768 dim, frozen)
- `google/siglip2-base-patch16-224`
- `google/siglip2-base-patch16-naflex`
- `google/siglip-so400m-patch14-384`
- `apple/MobileCLIP2-B`
- `<path-to-moonvit-config>` (native-resolution, 1152 dim, 27-layer, patch merge)
  - Activated by name containing "moonvit" or config with `merge_kernel_size`
  - `--ve_hidden_size 1152` must be set
  - See `locany/modeling_vit.py` for MoonViT architecture details

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

### Multi-Dataset Training (Data Recipe)

Train on multiple datasets simultaneously with per-dataset sampling weights and augmentation:

```bash
# Prepare RefCOCO/+/g
python train.py --action prepare_refcoco --coco_root ./data/coco --output_dir ./data/refcoco

# Prepare COCO detection
python train.py --action prepare_coco --image_dir ./data/coco --output_dir ./data/coco_detection

# Create a recipe JSON
cat > recipe.json << 'EOF'
{
    "refcoco": {
        "annotation": "data/refcoco/train.jsonl",
        "root": "data/coco/train2014",
        "repeat_time": 1.0,
        "data_augment": true
    },
    "detection_coco": {
        "annotation": "data/coco_detection/train.jsonl",
        "root": "data/coco/train2017",
        "repeat_time": 2.0,
        "data_augment": true
    }
}
EOF

# Train on both datasets (recipe overrides --train_data_path)
python train.py --action train \
  --data_recipe recipe.json \
  --output_dir ./outputs_multi \
  --num_epochs 5 \
  --per_device_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 3e-5
```

**Recipe fields:**
- `annotation` — path to JSONL file
- `root` — image root directory
- `repeat_time` — relative sampling weight (e.g. `2.0` = sampled twice as often)
- `data_augment` — enables random scale-crop augmentation per sample

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

### RefCOCO/+/g (Referring Expression Comprehension)
```bash
# Download all three RefCOCO variants from HuggingFace and combine into train/val
python train.py --action prepare_refcoco \
  --coco_root ./data/coco \
  --output_dir ./data/refcoco

# Train on combined RefCOCO/+/g (referring expression comprehension)
python train.py --action train \
  --train_data_path ./data/refcoco/train.jsonl \
  --image_dir ./data/coco/train2014 \
  --output_dir ./outputs_refcoco \
  --num_epochs 10 \
  --per_device_batch_size 1 \
  --learning_rate 5e-5 \
  --lora_r 64
```

### Object365 Dataset (Large-Scale Detection)

Objects365 v2 contains **365 categories** with ~2M images and ~30M bounding boxes. Good for pretraining on diverse real-world objects.

```bash
# Prepare Objects365 (requires ~180GB for full dataset)
python train.py --action prepare_object365 \
  --objects_root ./data/objects365 \
  --output_dir ./data/objects365_detection \
  --max_train 50000 --max_val 5000 \
  --max_patches 10

# Train on Objects365 detection
python train.py --action train \
  --train_data_path ./data/objects365_detection/train.jsonl \
  --image_dir ./data/objects365/images/train \
  --output_dir ./outputs_o365 \
  --num_epochs 5 \
  --per_device_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 3e-5 \
  --lora_r 64
```

**Notes:**
- Images come in 51 train patches (patch0–patch50) and 43 val patches (patch0–patch42), extracted from tar.gz archives
- Official KS3 download URLs can be slow outside China; consider OpenDataLab mirrors
- `--no-download-images` skips image download (use with pre-downloaded images)
- `--max-patches N` limits download for testing (e.g. `--max-patches 2` = only 2 patches per split)

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

## MoonViT Vision Encoder

MoonViT (from NVIDIA's EagleVLM) is a 27-layer, 1152-dim ViT with native-resolution support:

- **Native resolution**: Processes images at full input resolution (no fixed resize)
- **2D RoPE**: Rotary position embeddings in 2D space for variable-resolution generalization
- **Patch merging**: 2×2 kernel merges patches after encoding, producing (H/14/2)×(W/14/2) tokens with 4× channel width
- **Automatic VE detection**: Set `--ve_model <path-to-moonvit-config>` or any name containing "moonvit"; configs with `merge_kernel_size` are auto-detected

Activated by passing a MoonViT config path. The `MoonViTProjector` (LayerNorm + 4× channel → LLM hidden) handles the merged feature dimensions.

## Parallel Box Decoding (PBD/MTP)

PBD generates bounding boxes faster by predicting multiple tokens at once:

- **MTP mode** (`--generation_mode fast`): Uses non-causal attention within a prediction block (size=`--block_size 6`). All `block_size` tokens are decoded in parallel via `sample_tokens()` / `decode_bbox_avg()`.
- **Hybrid mode** (`--generation_mode hybrid`, default): Starts with MTP for box tokens, falls back to AR for text tokens. On malformed MTP output, resets to AR for that box.
- **AR mode** (`--generation_mode slow`): Standard token-by-token generation (backward-compatible).

PBD is only supported with batch_size=1 and is called via `model.generate_pbd()`.

## Sequence Packing

`PackedDetectionDataset` greedily concatenates multiple training samples into a single sequence up to `max_packed_tokens`, reducing padding waste:

- Tracks `sub_sample_lengths` to create proper per-sample causal attention masks
- Each sample gets its own `position_ids` (starting from 0), creating boundaries the mask logic detects
- `PackedDataCollator` stacks packed batches with the required metadata

Enable by wrapping your dataset: `PackedDetectionDataset(base_dataset, max_packed_tokens=2048)`.

## Key Design Decisions

- **Discrete tokens over regression**: Cross-entropy loss provides strong per-class supervision through the full model (LM head → LLM → projector → VE). Unlike MSE regression, gradients flow to every component.
- **LoRA on LLM** (r=64–128): The LLM must learn image-dependent hidden states at coordinate token positions. LoRA makes this feasible on a 4GB GPU.
- **Optional VE LoRA** (`--use_backbone_lora N`): Applies LoRA to vision encoder attention/MLP layers (r=N) for fine-grained visual adaptation.
- **Untied LM head**: `tie_word_embeddings=False` allows the LM head for new coordinate tokens to be trained independently from input embeddings.
- **Frozen VE**: Vision encoder stays frozen by default to save memory; projector and LoRA adapt visual features.
- **Dual generation modes**: PBD/MTP for faster inference on compatible models, AR fallback for backward compatibility.
- **MLP connector depth** (`--mlp_connector_layers N`): Supports 2+ layer projectors with LayerNorm for deeper visual-language alignment.

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

# Visualize (with labels)
from locany import visualize_boxes, parse_labels_and_boxes
label_boxes = parse_labels_and_boxes(result["text"])
labels = [lb[0] for lb in label_boxes]
visualize_boxes(image, result["boxes"], labels=labels, output_path="output.png")
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
  --use_backbone_lora N           Apply LoRA on VE (r=N, default: 0 = off)
  --mlp_connector_layers N        MLP projector depth (default: 2)
  --block_size N                  PBD block size (default: 6)
  --generation_mode {hybrid,fast,slow}  PBD generation mode (default: hybrid)
  --attn_implementation           {sdpa,flash_attention_2,eager}
  --torch_dtype                   {float32,float16,bfloat16}

Vision Encoders:
  google/siglip-base-patch16-224        (SigLIP, 224px, 768 dim, default)
  google/siglip2-base-patch16-224        (SigLIP2, 224px)
  google/siglip2-base-patch16-naflex     (SigLIP2 + FlexiViT)
  google/siglip-so400m-patch14-384       (So400m, 384px, 1152 dim)
  apple/MobileCLIP2-B                    (MobileCLIP)
  <moonvit-config-path>                  (MoonViT, native-res, 1152 dim, 27-layer)

Training:
  --output_dir, --num_epochs, --per_device_batch_size
  --learning_rate, --warmup_ratio, --weight_decay
  --bf16 / --no-bf16, --gradient_checkpointing
  --logging_steps, --save_steps, --save_total_limit
  --lr_scheduler_type, --max_grad_norm

Data:
  --train_data_path, --eval_data_path, --image_dir
  --data_recipe                  Path to multi-dataset recipe JSON
  --max_length
  --use_online_packing           Enable sequence packing in dataset (default: False)

Inference:
  --max_new_tokens, --temperature, --top_p
  --generation_mode {hybrid,fast,slow}  PBD vs AR generation

Prepare (RefCOCO):
  --coco_root, --ann_dir, --output_dir
  --splits, --num_train, --num_val
  --no-download, --no-combine

Prepare (COCO):
  --image_dir, --output_dir
  --max_train, --max_val, --no-download

Prepare (Objects365):
  --objects_root              Objects365 data directory (default: ./data/objects365)
  --output_dir                Output directory (default: ./data/objects365_detection)
  --splits                    Splits to process (default: train val)
  --max_train, --max_val      Limit images per split
  --no-download               Skip annotation download
  --no-download-images        Skip image download
  --max-patches               Limit patches downloaded per split

Actions:
  --action {train,inference,eval,create_sample,prepare_refcoco,prepare_coco,prepare_object365}
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers 4.45+ (for SigLIP2 support)
- accelerate, Pillow, torchvision
- peft (for LoRA)
- Optional: deepspeed, tensorboard

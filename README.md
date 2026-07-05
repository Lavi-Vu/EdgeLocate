# EdgeLocate — <1B LocateAnything with Discrete Coordinate Tokens

A <1B variant of NVIDIA's LocateAnything (EagleVL) using **discrete coordinate tokens** (`<0>`–`<1000>`) predicted via cross-entropy loss through the LM head. Supports dual vision encoders (SigLIP / MoonViT) and Parallel Box Decoding (PBD/MTP) for faster inference.

## Installation

```bash
pip install torch transformers accelerate pillow torchvision peft safetensors tensorboard
```

Python 3.10+, PyTorch 2.0+, transformers 4.45+.

## Data Preparation

### Synthetic data (quick test)
```bash
python train.py --action create_sample --train_data_path ./data.jsonl --num_samples 200 --max_boxes_per_image 8
```

### COCO detection
Downloads COCO 2017 images + annotations, converts to JSONL with `<ref>cat</ref><box><d1><d2><d3><d4></box>` format:
```bash
# Prepare train.jsonl and val.jsonl
python train.py --action prepare_coco \
  --image_dir ./data/coco \
  --output_dir ./data/coco_detection \
  --max_train 50000 --max_val 1000
```

### RefCOCO/+/g (referring expression comprehension)
Downloads from HuggingFace and combines all three variants:
```bash
python train.py --action prepare_refcoco \
  --coco_root ./data/coco \
  --output_dir ./data/refcoco
```

### Objects365 (large-scale detection)
365 categories, ~2M images, ~30M boxes. Requires ~180GB:
```bash
python train.py --action prepare_object365 \
  --objects_root ./data/objects365 \
  --output_dir ./data/objects365_detection \
  --max_train 50000 --max_val 5000 \
  --max_patches 10
```

## Training

### Basic training
```bash
python train.py --action train \
  --train_data_path ./data/coco_detection/train.jsonl \
  --image_dir ./data/coco/train2017 \
  --output_dir ./outputs_coco \
  --num_epochs 5 \
  --gradient_accumulation_steps 8 \
  --per_device_batch_size 1 \
  --learning_rate 3e-5 \
  --lora_r 64
```

### Multi-dataset training (data recipe)
Train on multiple datasets with per-dataset sampling weights and augmentation:
```bash
cat > recipe.json << 'EOF'
{
  "refcoco": {
    "annotation": "data/refcoco/train.jsonl",
    "root": "data/coco/train2014",
    "repeat_time": 1.0,
    "data_augment": true
  },
  "coco_detection": {
    "annotation": "data/coco_detection/train.jsonl",
    "root": "data/coco/train2017",
    "repeat_time": 2.0,
    "data_augment": true
  }
}
EOF

python train.py --action train \
  --data_recipe recipe.json \
  --output_dir ./outputs_multi \
  --num_epochs 5 \
  --gradient_accumulation_steps 8
```

Recipe fields:

| Field | Type | Description |
|---|---|---|
| `annotation` | string | Path to JSONL file |
| `root` | string | Image root directory |
| `repeat_time` | float | Relative sampling weight (e.g. `2.0` = sampled twice as often) |
| `data_augment` | bool | Random long-edge resize augmentation (50% prob, target [640, 2560]) |

### MoonViT training
```bash
python train.py --action train \
  --ve_model <path-to-MoonViT-SO-400M> \
  --ve_hidden_size 1152 \
  --train_data_path ./data.jsonl \
  --image_dir . \
  --output_dir ./outputs_moonvit \
  --num_epochs 10 \
  --per_device_batch_size 1
```

### SigLIP2 training
```bash
python train.py --action train \
  --ve_model google/siglip2-base-patch16-224 \
  --train_data_path ./data.jsonl \
  --image_dir . \
  --output_dir ./outputs_siglip2
```

### Low-memory training (4GB GPU)
```bash
python train.py --action train \
  --lora_r 64 \
  --gradient_checkpointing \
  --per_device_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --bf16
```

### Sequence packing
Reduce padding waste by concatenating multiple samples into one sequence:
```bash
python train.py --action train \
  --packing \
  --max_length 2048 \
  --train_data_path ./data.jsonl
```

Each sample gets its own `position_ids` starting from 0; the model creates per-sample causal boundaries via `sub_sample_lengths`.

## Evaluation

### Standard benchmark
```bash
python evaluate.py \
  --model_dir ./outputs_coco \
  --data ./data/coco_detection/val.jsonl \
  --image_dir ./data/coco/val2017 \
  --batch_size 8 \
  --max_samples 500 \
  --output ./results.json
```

### PBD mode evaluation
```bash
# Hybrid (PBD with AR fallback, default)
python evaluate.py --model_dir ./outputs --data ./val.jsonl --image_dir . --mode hybrid

# Pure PBD (no fallback)
python evaluate.py --model_dir ./outputs --data ./val.jsonl --image_dir . --mode fast

# Pure autoregressive
python evaluate.py --model_dir ./outputs --data ./val.jsonl --image_dir . --mode slow
```

PBD modes (`fast`/`hybrid`) work with any vision encoder; `slow` is pure autoregressive for backward compatibility.

### Visualize predictions
```bash
python evaluate.py --model_dir ./outputs --data ./val.jsonl --image_dir . \
  --max_samples 100 --visualize 10
```

Saves `vis/vis_{img_id}_{filename}.jpg` with predicted boxes (red) and ground truth (green) overlaid.

## Inference

```bash
python infer.py \
  --model_dir ./outputs \
  --image path/to/image.jpg \
  --prompt "Locate all the instances that matches the following description: all objects."

# With PBD mode
python infer.py --model_dir ./outputs --image img.jpg --prompt "detect person" --mode hybrid

# Programmatic API
python -c "
from locany import ModelConfig, load_model_from_dir, DetectionInferenceEngine, InferenceConfig
from transformers import AutoTokenizer
from PIL import Image

model = load_model_from_dir('./outputs', AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')).cuda().eval()
engine = DetectionInferenceEngine(model, tokenizer, InferenceConfig(mode='hybrid'))
result = engine.predict(Image.open('img.jpg').convert('RGB'), 'detect all objects.')
print(result['boxes'])  # [[x1,y1,x2,y2], ...] in pixel coords

# Visualize with labels
from locany import visualize_boxes, parse_labels_and_boxes
label_boxes = parse_labels_and_boxes(result['text'])
visualize_boxes(image, result['boxes'], labels=[lb for lb,_ in label_boxes], output_path='out.png')
"
```

## Generation Modes

Controlled by `--mode` flag (`--generation_mode` in train.py's model group):

| Mode | Description | Speed | Fallback |
|---|---|---|---|
| `hybrid` (default) | MTP for box tokens, AR for text. On malformed MTP output, falls back to AR. | Fast | Yes |
| `fast` | Pure MTP — all tokens predicted in blocks of `block_size=6`. | Fastest | No |
| `slow` | Standard autoregressive token-by-token via `model.generate()`. | Baseline | N/A |

PBD dispatch rule: `mode != 'slow'` → `model.generate_pbd()`. Otherwise → `model.generate()`.

## Data Format

ShareGPT-style JSONL with inline `<ref>` and `<box>` tokens:

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

- **Roles**: `human`/`gpt` (also `user`/`assistant`)
- **Prompt**: Categories separated by `</c>`, stripped of `<|image|>\n` prefix at inference
- **Response**: One `<ref>label</ref><box><d1><d2><d3><d4></box>` per instance
- **Coordinates**: Integer tokens `<0>`–`<1000>`, denormalized to pixel space by `coord * img_dim / 1000`
- **Image path**: Relative to `image_dir` or absolute

## Token Scheme

| Token | ID | Usage |
|---|---|---|
| `<\|image\|>` | 151665 | Image anchor in user text |
| `<box>` | 151666 | Box coordinate start |
| `</box>` | 151667 | Box coordinate end |
| `<ref>` | 151668 | Label/object name start |
| `</ref>` | 151669 | Label/object name end |
| `<0>`–`<1000>` | 151670–152670 | 1001 quantized coordinate bins |
| `<null>` | 152671 | No detection |
| `<text_mask>` | 152672 | MTP mask placeholder |

Total vocabulary: 152673 tokens (base Qwen2.5 + 1001 coord + 8 special).

## Model Components

| Component | Model | Params (SigLIP) | Params (MoonViT) | Default |
|---|---|---|---|---|
| Vision Encoder | SigLIP-Base-P16-224 / MoonViT-SO-400M | 92.88M | 408.15M | Frozen |
| VE LoRA | Optional backbone LoRA (`--use_backbone_lora N`) | 0 | ~6M | Off |
| MLP Projector | 2–4 layer Linear+GELU (VE dim → LLM dim 896) | ~1.5M | ~1.8M | Trainable |
| LLM | Qwen2.5-0.5B-Instruct + LoRA (r=128) | 494.03M | 494.03M | LoRA only |
| LM Head | Untied output projection (896 → vocab) | 0.14M | 0.14M | Trainable |
| **Total** | | **~589M** | **~904M** | **~37–44M trainable** |

## All CLI Flags

### Model

| Flag | Default | Description |
|---|---|---|
| `--llm_model` | `Qwen/Qwen2.5-0.5B-Instruct` | LLM backbone |
| `--ve_model` | `google/siglip-base-patch16-224` | Vision encoder |
| `--ve_hidden_size` | `768` | VE output dim (1152 for MoonViT) |
| `--llm_hidden_size` | `896` | LLM hidden dim |
| `--max_boxes` | `32` | Max boxes per image |
| `--freeze_llm` | `False` | Freeze entire LLM |
| `--freeze_vision_encoder` | `True` | Freeze VE weights |
| `--freeze_mlp` | `False` | Freeze MLP projector |
| `--vision_select_layer` | `-1` | Which VE layer to use (last) |
| `--attn_implementation` | `sdpa` | Attention backend: `sdpa`, `flash_attention_2`, `eager` |
| `--torch_dtype` | `bfloat16` | Model dtype |
| `--use_lora` / `--no-lora` | `True` | Enable LoRA on LLM |
| `--lora_r` | `128` | LoRA rank |
| `--lora_alpha` | `256` | LoRA alpha |
| `--use_backbone_lora` | `0` | LoRA rank on VE (0 = off) |
| `--mlp_connector_layers` | `2` | MLP projector depth |
| `--block_size` | `6` | PBD/MTP block size |
| `--generation_mode` | `hybrid` | PBD mode: `fast`, `hybrid`, `slow` |

### Training

| Flag | Default | Description |
|---|---|---|
| `--output_dir` | `./outputs` | Save directory |
| `--num_epochs` | `3` | Training epochs |
| `--per_device_batch_size` | `4` | Batch size per GPU |
| `--gradient_accumulation_steps` | `1` | Gradient accumulation |
| `--learning_rate` | `2e-5` | Peak LR |
| `--warmup_ratio` | `0.03` | LR warmup fraction |
| `--weight_decay` | `0.1` | AdamW weight decay |
| `--bf16` / `--no-bf16` | `True` | BFloat16 mixed precision |
| `--gradient_checkpointing` | `True` | Activation checkpointing |
| `--deepspeed` | `None` | DeepSpeed config path |
| `--logging_steps` | `10` | Log interval |
| `--save_steps` | `500` | Save checkpoint interval |
| `--eval_steps` | `500` | Evaluation interval |
| `--save_total_limit` | `3` | Max checkpoints to keep |
| `--lr_scheduler_type` | `cosine` | LR schedule |
| `--max_grad_norm` | `1.0` | Gradient clipping |
| `--packing` | `False` | Enable sequence packing |
| `--seed` | `42` | Random seed |

### Data

| Flag | Default | Description |
|---|---|---|
| `--train_data_path` | `""` | Training JSONL path |
| `--eval_data_path` | `""` | Evaluation JSONL path |
| `--image_dir` | `""` | Image root directory |
| `--data_recipe` | `None` | Multi-dataset recipe JSON path |
| `--max_length` | `2048` | Max sequence length |

### Inference (`train.py` / `evaluate.py`)

| Flag | Default | Description |
|---|---|---|
| `--mode` | `hybrid` | Generation mode: `fast`, `hybrid`, `slow` |
| `--max_new_tokens` | `512` | Max generated tokens |
| `--max_new_boxes` | `32` | Max boxes to decode |
| `--temperature` | `0.0` | Sampling temperature (0 = greedy) |
| `--top_p` | `1.0` | Nucleus sampling |
| `--keep_k_avg` | `4` | PBD coordinate averaging candidates |
| `--confidence_threshold` | `0.0` | Min confidence (reserved) |
| `--batch_size` | `8` | Eval batch size (predict_batch) |
| `--max_samples` | `None` | Limit eval samples |
| `--iou_threshold` | `0.5` | IoU threshold for metrics |
| `--output` | `None` | Save results JSON path |
| `--visualize` | `0` | Save N prediction visualization images |

### Data Preparation

| Flag | Default | Description |
|---|---|---|
| `--num_samples` | `100` | Synthetic data count |
| `--max_boxes_per_image` | `8` | Synthetic max boxes |
| `--no-download` | `False` | Skip downloads |
| `--max_train` | `None` | Limit train images (COCO/O365) |
| `--max_val` | `None` | Limit val images (COCO/O365) |
| `--coco_root` | `./data/coco` | COCO directory |
| `--ann_dir` | `""` | COCO annotation directory |
| `--splits` | `train val` | Splits to process |
| `--num_train` | `None` | RefCOCO train limit |
| `--num_val` | `None` | RefCOCO val limit |
| `--no-combine` | `False` | Don't merge RefCOCO variants |
| `--objects_root` | `./data/objects365` | Objects365 directory |
| `--no-download-images` | `False` | Skip O365 image download |
| `--max-patches` | `None` | Limit O365 patches |

### Actions

| Action | Description |
|---|---|
| `train` | Train model |
| `inference` | Run inference on image_dir images |
| `eval` | Evaluate on dataset (legacy) |
| `create_sample` | Generate synthetic JSONL |
| `prepare_coco` | Convert COCO to JSONL |
| `prepare_refcoco` | Convert RefCOCO/+/g to JSONL |
| `prepare_object365` | Convert Objects365 to JSONL |

## Supported Vision Encoders

| Model | Resolution | Dim | Notes |
|---|---|---|---|
| `google/siglip-base-patch16-224` | 224×224 | 768 | Default, frozen |
| `google/siglip2-base-patch16-224` | 224×224 | 768 | SigLIP2 (Linear patch_embed) |
| `google/siglip2-base-patch16-naflex` | variable | 768 | FlexiViT patches |
| `google/siglip-so400m-patch14-384` | 384×384 | 1152 | Larger SigLIP |
| `apple/MobileCLIP2-B` | variable | | Mobile-optimized |
| `moonshotai/MoonViT-SO-400M` | native | 1152 | PBD-capable, 2D RoPE, patch merge |

## Output Metrics

| Metric | Description |
|---|---|
| `num_samples` | Total evaluated images |
| `mean_iou` | Mean IoU across all predictions |
| `AP` | Mean AP @ IoU 0.50:0.05:0.95 |
| `AP@0.50` | PASCAL VOC standard |
| `AP@0.75` | Strict localization |
| `AP@0.90` | Near-perfect localization |
| `Precision` | Mean precision across thresholds |
| `Recall` | Mean recall across thresholds |
| `F1` | Mean F1 across thresholds |

## Epoch Visualization

During training, `epoch_vis/epoch_N.jpg` is saved at start (epoch 0) and after each epoch, showing 8 random augmentations per sampled image (baseline + 7 augmented variants). Enabled automatically; no flag needed. See `TrainVisCallback` in `training.py`.

# EdgeLocate — <1B LocateAnything with Discrete Coordinate Tokens

A <1B variant of NVIDIA's LocateAnything using **discrete coordinate tokens** (`<0>`–`<1000>`) predicted via cross-entropy loss through the LM head, instead of a regression head. Supports dual vision encoders (SigLIP / MoonViT) and Parallel Box Decoding (PBD/MTP).

## Quick Start

```bash
pip install torch transformers accelerate pillow torchvision peft safetensors tensorboard
```

### Create synthetic data & train
```bash
python train.py --action create_sample --train_data_path ./data.jsonl --num_samples 200
python train.py --action train --train_data_path ./data.jsonl --image_dir . --output_dir ./outputs --num_epochs 20
```

### Train on COCO
```bash
python train.py --action prepare_coco --image_dir ./data/coco --output_dir ./data/coco_detection --max_train 50000 --max_val 1000
python train.py --action train --train_data_path ./data/coco_detection/train.jsonl --image_dir ./data/coco/train2017 --output_dir ./outputs_coco --num_epochs 5 --gradient_accumulation_steps 8
```

### Multi-dataset training (data recipe)
```bash
python train.py --action prepare_refcoco --coco_root ./data/coco --output_dir ./data/refcoco
cat > recipe.json << 'EOF'
{"coco": {"annotation": "data/coco_detection/train.jsonl", "root": "data/coco/train2017", "repeat_time": 1.0, "data_augment": true}}
EOF
python train.py --action train --data_recipe recipe.json --output_dir ./outputs_multi --num_epochs 5
```

### Evaluate
```bash
python evaluate.py --model_dir ./outputs_coco --data ./data/coco_detection/val.jsonl --image_dir ./data/coco/val2017 --batch_size 8 --max_samples 500 --output ./results.json
```

### Evaluate with PBD mode
```bash
python evaluate.py --model_dir ./outputs_coco --data ./data/coco_detection/val.jsonl --image_dir ./data/coco/val2017 --mode hybrid --max_samples 500
```

### Visualize predictions during eval
```bash
python evaluate.py --model_dir ./outputs_coco --data ./data/coco_detection/val.jsonl --image_dir ./data/coco/val2017 --max_samples 500 --visualize 10
```

## Data Format

ShareGPT-style JSONL with `<ref>` and `<box>` tokens in the GPT response:

```json
{
  "image": "coco/train2017/000001.jpg",
  "conversations": [
    {"from": "human", "value": "Locate all the instances that matches the following description: car</c>person."},
    {"from": "gpt", "value": "<ref>car</ref><box><120><200><450><500></box><ref>person</ref><box><50><100><200><600></box>"}
  ]
}
```

Coordinates are quantized to `[0, 1000]` integer bins, normalized by `image_dim / 1000`.

## Model Components

| Component | Details | Params (SigLIP) | Freeze |
|---|---|---|---|
| Vision Encoder | SigLIP-Base-P16-224 / MoonViT-SO-400M | 92.88M / 408.15M | Frozen |
| Projector | 2–4 layer MLP (VE dim → LLM dim 896) | ~1.5M | Trainable |
| LLM | Qwen2.5-0.5B-Instruct + LoRA (r=128) | 494.03M | LoRA only |
| LM Head | Untied output projection | 0.14M | Trainable |

See [ARCHITECTURE.md](ARCHITECTURE.md) for full details.

## Supported Vision Encoders

| Model | Resolution | Dim | Notes |
|---|---|---|---|
| `google/siglip-base-patch16-224` | 224×224 | 768 | Default |
| `google/siglip2-base-patch16-224` | 224×224 | 768 | |
| `google/siglip2-base-patch16-naflex` | variable | 768 | FlexiViT |
| `google/siglip-so400m-patch14-384` | 384×384 | 1152 | |
| `moonshotai/MoonViT-SO-400M` | native | 1152 | PBD-capable |

## CLI Arguments

Full options via `--help`:

```
--action {train,inference,eval,create_sample,prepare_refcoco,prepare_coco,prepare_object365}
```

Key groups: `Model`, `Training`, `Data`, `Inference`, `Prepare`. Examples:

```bash
# Use MoonViT with PBD
python train.py --action train --ve_model .../MoonViT-SO-400M --ve_hidden_size 1152 --generation_mode hybrid

# Use SigLIP2
python train.py --action train --ve_model google/siglip2-base-patch16-224

# Low-memory training
python train.py --action train --lora_r 64 --gradient_checkpointing --per_device_batch_size 1
```

## Data Recipes (Multi-Dataset)

| Field | Description |
|---|---|
| `annotation` | Path to JSONL |
| `root` | Image root directory |
| `repeat_time` | Relative sampling weight |
| `data_augment` | Enable random long-edge resize augmentation |

## Output Metrics

| Metric | Description |
|---|---|
| `AP` | Mean Average Precision @ IoU 0.50:0.95 |
| `AP@0.50` | PASCAL VOC standard |
| `AP@0.75` | Strict localization |
| `F1` | Mean F1 across thresholds |
| `Precision`/`Recall` | Mean across thresholds |

## Requirements

Python 3.10+, PyTorch 2.0+, transformers 4.45+, accelerate, peft, Pillow, torchvision.

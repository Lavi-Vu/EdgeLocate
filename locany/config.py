import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ModelConfig:
    llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    ve_model: str = "google/siglip-base-patch16-224"
    ve_hidden_size: int = 768
    llm_hidden_size: int = 896
    llm_max_length: int = 2048
    max_boxes: int = 32
    coord_bins: int = 1001
    freeze_llm: bool = False
    freeze_vision_encoder: bool = True
    freeze_mlp: bool = False
    vision_select_layer: int = -1
    use_flash_attn: bool = False
    attn_implementation: str = "sdpa"
    torch_dtype: str = "bfloat16"
    use_lora: bool = True
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05

    def to_dict(self) -> Dict:
        return {
            "llm_model": self.llm_model,
            "ve_model": self.ve_model,
            "ve_hidden_size": self.ve_hidden_size,
            "llm_hidden_size": self.llm_hidden_size,
            "llm_max_length": self.llm_max_length,
            "max_boxes": self.max_boxes,
            "coord_bins": self.coord_bins,
            "freeze_llm": self.freeze_llm,
            "freeze_vision_encoder": self.freeze_vision_encoder,
            "freeze_mlp": self.freeze_mlp,
            "vision_select_layer": self.vision_select_layer,
            "use_flash_attn": self.use_flash_attn,
            "attn_implementation": self.attn_implementation,
            "torch_dtype": self.torch_dtype,
            "use_lora": self.use_lora,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
        }

    @staticmethod
    def from_dict(d: Dict) -> "ModelConfig":
        return ModelConfig(**{k: v for k, v in d.items() if k in ModelConfig.__dataclass_fields__})


@dataclass
class TrainingConfig:
    output_dir: str = "./outputs"
    num_epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.1
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    deepspeed: Optional[str] = None
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    block_size: int = 2048
    packing: bool = False
    seed: int = 42


@dataclass
class DataConfig:
    train_data_path: str = ""
    eval_data_path: str = ""
    image_dir: str = ""
    data_recipe: Optional[str] = None
    max_length: int = 2048
    image_size: Tuple[int, int] = (224, 224)


@dataclass
class InferenceConfig:
    mode: str = "fast"
    max_new_boxes: int = 32
    confidence_threshold: float = 0.0
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0


def load_data_recipe(path: str) -> Dict:
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="LocateAnything PBD Training/Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model args
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--llm_model", default="Qwen/Qwen2.5-0.5B-Instruct", help="LLM backbone")
    model_group.add_argument("--ve_model", default="google/siglip-base-patch16-224", help="Vision encoder")
    model_group.add_argument("--ve_hidden_size", type=int, default=768)
    model_group.add_argument("--llm_hidden_size", type=int, default=896)
    model_group.add_argument("--max_boxes", type=int, default=32, help="Maximum boxes per image")
    model_group.add_argument("--freeze_llm", action="store_true", default=False)
    model_group.add_argument("--no-freeze_llm", action="store_false", dest="freeze_llm")
    model_group.add_argument("--freeze_vision_encoder", action="store_true", default=True)
    model_group.add_argument("--no-freeze_vision_encoder", action="store_false", dest="freeze_vision_encoder")
    model_group.add_argument("--freeze_mlp", action="store_true", default=False)
    model_group.add_argument("--vision_select_layer", type=int, default=-1)
    model_group.add_argument("--attn_implementation", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    model_group.add_argument("--torch_dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    model_group.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA on LLM")
    model_group.add_argument("--no-lora", action="store_false", dest="use_lora")
    model_group.add_argument("--lora_r", type=int, default=128)
    model_group.add_argument("--lora_alpha", type=int, default=256)

    # Training args
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--output_dir", default="./outputs")
    train_group.add_argument("--num_epochs", type=int, default=3)
    train_group.add_argument("--per_device_batch_size", type=int, default=4)
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=1)
    train_group.add_argument("--learning_rate", type=float, default=2e-5)
    train_group.add_argument("--warmup_ratio", type=float, default=0.03)
    train_group.add_argument("--weight_decay", type=float, default=0.1)
    train_group.add_argument("--bf16", action="store_true", default=True)
    train_group.add_argument("--no-bf16", action="store_false", dest="bf16")
    train_group.add_argument("--gradient_checkpointing", action="store_true", default=True)
    train_group.add_argument("--deepspeed", default=None, help="Path to DeepSpeed config JSON")
    train_group.add_argument("--logging_steps", type=int, default=10)
    train_group.add_argument("--save_steps", type=int, default=500)
    train_group.add_argument("--eval_steps", type=int, default=500)
    train_group.add_argument("--save_total_limit", type=int, default=3)
    train_group.add_argument("--lr_scheduler_type", default="cosine")
    train_group.add_argument("--max_grad_norm", type=float, default=1.0)
    train_group.add_argument("--block_size", type=int, default=2048)
    train_group.add_argument("--packing", action="store_true", default=False)
    train_group.add_argument("--seed", type=int, default=42)

    # Data args
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--train_data_path", default="", help="Path to training JSONL")
    data_group.add_argument("--eval_data_path", default="", help="Path to eval JSONL")
    data_group.add_argument("--image_dir", default="", help="Image directory")
    data_group.add_argument("--data_recipe", default=None, help="Path to data recipe JSON")
    data_group.add_argument("--max_length", type=int, default=2048)

    # Inference args
    inf_group = parser.add_argument_group("Inference")
    inf_group.add_argument("--mode", default="fast", choices=["fast", "hybrid", "slow"])
    inf_group.add_argument("--max_new_boxes", type=int, default=32)
    inf_group.add_argument("--confidence_threshold", type=float, default=0.0)
    inf_group.add_argument("--max_new_tokens", type=int, default=512)
    inf_group.add_argument("--temperature", type=float, default=0.0)
    inf_group.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples for create_sample action")
    parser.add_argument("--max_boxes_per_image", type=int, default=8, help="Max boxes per image for create_sample")
    parser.add_argument("--no-download", action="store_true", help="Skip download for prepare actions")
    parser.add_argument("--max_train", type=int, default=None, help="Max train images for prepare_coco")
    parser.add_argument("--max_val", type=int, default=None, help="Max val images for prepare_coco")

    # Prepare args
    parser.add_argument("--coco_root", default="./data/coco", help="COCO image directory (prepare_refcoco)")
    parser.add_argument("--ann_dir", default="", help="Pre-downloaded annotation directory (prepare_refcoco)")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to process (prepare_refcoco)")
    parser.add_argument("--num_train", type=int, default=None, help="Max train samples per variant (prepare_refcoco)")
    parser.add_argument("--num_val", type=int, default=None, help="Max val samples per variant (prepare_refcoco)")
    parser.add_argument("--no-combine", action="store_true", help="Skip combining variants (prepare_refcoco)")

    # Action mode
    parser.add_argument("--action", default="train", choices=["train", "inference", "eval", "create_sample", "prepare_refcoco", "prepare_coco"],
                        help="Action to perform")

    args = parser.parse_args()

    model_cfg = ModelConfig(
        llm_model=args.llm_model,
        ve_model=args.ve_model,
        ve_hidden_size=args.ve_hidden_size,
        llm_hidden_size=args.llm_hidden_size,
        max_boxes=args.max_boxes,
        freeze_llm=args.freeze_llm,
        freeze_vision_encoder=args.freeze_vision_encoder,
        freeze_mlp=args.freeze_mlp,
        vision_select_layer=args.vision_select_layer,
        use_flash_attn=(args.attn_implementation == "flash_attention_2"),
        attn_implementation=args.attn_implementation,
        torch_dtype=args.torch_dtype,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    train_cfg = TrainingConfig(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        block_size=args.block_size,
        packing=args.packing,
        seed=args.seed,
    )

    data_cfg = DataConfig(
        train_data_path=args.train_data_path,
        eval_data_path=args.eval_data_path,
        image_dir=args.image_dir,
        data_recipe=args.data_recipe,
        max_length=args.max_length,
    )

    infer_cfg = InferenceConfig(
        mode=args.mode,
        max_new_boxes=args.max_new_boxes,
        confidence_threshold=args.confidence_threshold,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    return model_cfg, train_cfg, data_cfg, infer_cfg, args.action, args.no_download, args.max_train, args.max_val, args

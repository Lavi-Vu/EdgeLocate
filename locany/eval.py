import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


def compute_iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_precision_recall(
    pred_boxes: List[List[float]],
    gt_boxes: List[List[float]],
    iou_threshold: float = 0.5,
) -> Tuple[float, float, float]:
    if not pred_boxes and not gt_boxes:
        return 1.0, 1.0, 1.0
    if not pred_boxes:
        return 0.0, 0.0, 0.0
    if not gt_boxes:
        return 0.0, 0.0, 0.0
    matched_gt = set()
    true_positives = 0
    for pred in pred_boxes:
        best_iou = 0
        best_idx = -1
        for j, gt in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_iou >= iou_threshold:
            true_positives += 1
            matched_gt.add(best_idx)
    precision = true_positives / len(pred_boxes) if pred_boxes else 0.0
    recall = true_positives / len(gt_boxes) if gt_boxes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_ap(
    pred_boxes_by_image: Dict[int, List[List[float]]],
    gt_boxes_by_image: Dict[int, List[List[float]]],
    iou_threshold: float = 0.5,
) -> float:
    all_preds = []
    n_gt = 0
    for img_id in gt_boxes_by_image:
        gts = gt_boxes_by_image[img_id]
        n_gt += len(gts)
        preds = pred_boxes_by_image.get(img_id, [])
        matched = set()
        for pred in preds:
            best_iou = 0
            best_idx = -1
            for j, gt in enumerate(gts):
                if j in matched:
                    continue
                iou = compute_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            is_tp = best_iou >= iou_threshold
            if is_tp:
                matched.add(best_idx)
            all_preds.append((is_tp, best_iou))
    if not all_preds:
        return 0.0
    all_preds.sort(key=lambda x: -x[1])
    tp = np.cumsum([1 if p[0] else 0 for p in all_preds])
    fp = np.cumsum([1 if not p[0] else 0 for p in all_preds])
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(n_gt, 1)
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        p = np.max(prec[rec >= t]) if np.any(rec >= t) else 0.0
        ap += p / 11
    return ap


def compute_coco_ap(
    pred_boxes_by_image: Dict[int, List[List[float]]],
    gt_boxes_by_image: Dict[int, List[List[float]]],
    iou_thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    results = {}
    all_aps = []
    all_precs = []
    all_recs = []
    all_f1s = []

    for iou_th in iou_thresholds:
        ap = compute_ap(pred_boxes_by_image, gt_boxes_by_image, iou_th)
        all_aps.append(ap)

        precs = []
        recs = []
        for img_id in gt_boxes_by_image:
            preds = pred_boxes_by_image.get(img_id, [])
            gts = gt_boxes_by_image[img_id]
            p, r, f = compute_precision_recall(preds, gts, iou_th)
            precs.append(p)
            recs.append(r)
        mp = np.mean(precs) if precs else 0.0
        mr = np.mean(recs) if recs else 0.0
        mf = 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0
        all_precs.append(mp)
        all_recs.append(mr)
        all_f1s.append(mf)

        results[f"AP@{iou_th:.2f}"] = ap
        results[f"Precision@{iou_th:.2f}"] = mp
        results[f"Recall@{iou_th:.2f}"] = mr
        results[f"F1@{iou_th:.2f}"] = mf

    results["AP"] = np.mean(all_aps)
    results["Precision"] = np.mean(all_precs)
    results["Recall"] = np.mean(all_recs)
    results["F1"] = np.mean(all_f1s)
    return results


def _resolve_image_path(image_path: str, image_dir: str) -> Optional[str]:
    resolved = image_path if os.path.isabs(image_path) else os.path.join(image_dir, image_path)
    if not os.path.exists(resolved):
        resolved = os.path.join(image_dir, os.path.basename(image_path))
    return resolved if os.path.exists(resolved) else None


def run_benchmark(
    model,
    tokenizer,
    dataset,
    image_dir: str,
    max_samples: Optional[int] = None,
    iou_threshold: float = 0.5,
    batch_size: int = 8,
) -> Dict[str, float]:
    from .config import InferenceConfig
    from .inference import DetectionInferenceEngine
    from .utils import parse_boxes_from_text

    inf_cfg = InferenceConfig(max_new_tokens=512)
    engine = DetectionInferenceEngine(model, tokenizer, inf_cfg)

    pred_boxes_by_image = {}
    gt_boxes_by_image = {}
    all_ious = []
    all_precisions = []
    all_recalls = []

    from PIL import Image

    valid_indices = []
    for i in range(len(dataset)):
        if max_samples and i >= max_samples:
            break
        raw = dataset.data[i]
        image_path = raw.get("image", "")
        resolved = _resolve_image_path(image_path, image_dir)
        if resolved is None:
            continue
        gt_text = None
        for conv in raw.get("conversations", []):
            if conv.get("from") in ("gpt", "assistant"):
                gt_text = conv["value"]
                break
        if gt_text and parse_boxes_from_text(gt_text):
            valid_indices.append(i)

    num_samples = len(valid_indices)
    if max_samples:
        num_samples = min(num_samples, max_samples)
        valid_indices = valid_indices[:num_samples]

    iterator = tqdm(range(0, num_samples, batch_size), desc="Benchmark")
    for start_idx in iterator:
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = valid_indices[start_idx:end_idx]

        batch_images = []
        batch_gt_boxes = []
        batch_ids = []

        for idx in batch_indices:
            raw = dataset.data[idx]
            resolved = _resolve_image_path(raw["image"], image_dir)
            image = Image.open(resolved).convert("RGB")
            batch_images.append(image)
            batch_ids.append(idx)

            gt_text = None
            for conv in raw.get("conversations", []):
                if conv.get("from") in ("gpt", "assistant"):
                    gt_text = conv["value"]
                    break
            gt_boxes = parse_boxes_from_text(gt_text or "")
            batch_gt_boxes.append(gt_boxes)

        human_text = ""
        raw0 = dataset.data[valid_indices[0]]
        for conv in raw0.get("conversations", []):
            if conv.get("from") in ("human", "user"):
                human_text = conv["value"]
                break
        prompt = _make_prompt(human_text)

        batch_results = engine.predict_batch(
            batch_images, [prompt] * len(batch_images), batch_size=len(batch_images)
        )

        for j, result in enumerate(batch_results):
            pred_boxes = result["boxes"]
            gt_boxes = batch_gt_boxes[j]
            img_id = batch_ids[j]

            pred_boxes_by_image[img_id] = pred_boxes
            gt_boxes_by_image[img_id] = gt_boxes

            if pred_boxes and gt_boxes:
                box_ious = [compute_iou(p, g) for p in pred_boxes for g in gt_boxes]
                all_ious.append(max(box_ious))
                p, r, _ = compute_precision_recall(pred_boxes, gt_boxes, iou_threshold)
                all_precisions.append(p)
                all_recalls.append(r)
            elif not pred_boxes and gt_boxes:
                all_precisions.append(0.0)
                all_recalls.append(0.0)
            elif pred_boxes and not gt_boxes:
                all_precisions.append(0.0)
                all_recalls.append(0.0)

        iterator.set_postfix({"samples": min(end_idx, num_samples)})

    results = {
        "num_samples": len(all_precisions),
        "mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
    }

    coco_aps = compute_coco_ap(pred_boxes_by_image, gt_boxes_by_image)
    results.update(coco_aps)

    return results


def _make_prompt(human_text: str) -> str:
    if not human_text or human_text.startswith("<|image|>"):
        return "Locate all the instances that matches the following description: all objects."
    return human_text.replace("<|image|>\n", "")


def evaluate_model(
    model,
    tokenizer,
    eval_dataset,
    iou_threshold: float = 0.5,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Legacy evaluation entry point used by train.py."""
    from .config import InferenceConfig
    from .inference import DetectionInferenceEngine
    from .utils import parse_boxes_from_text
    from PIL import Image
    import numpy as np
    import transformers

    inf_cfg = InferenceConfig(max_new_tokens=512)
    engine = DetectionInferenceEngine(model, tokenizer, inf_cfg)
    device = next(model.parameters()).device

    all_ious = []
    all_precisions = []
    all_recalls = []

    from tqdm import tqdm
    iterator = tqdm(range(len(eval_dataset)))
    for i in iterator:
        if max_samples and i >= max_samples:
            break
        sample = eval_dataset[i]
        pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)

        label_ids = sample["labels"]
        label_text = tokenizer.decode(label_ids.tolist(), skip_special_tokens=False)
        gt_boxes = parse_boxes_from_text(label_text)

        gen_config = transformers.GenerationConfig(
            max_new_tokens=512, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config,
        )
        full_ids = generated.sequences if hasattr(generated, "sequences") else generated
        text_out = tokenizer.decode(full_ids[0], skip_special_tokens=False)
        pred_boxes = parse_boxes_from_text(text_out)

        if gt_boxes and pred_boxes:
            box_ious = [compute_iou(p, g) for p in pred_boxes for g in gt_boxes]
            all_ious.append(max(box_ious))
            precision, recall, _ = compute_precision_recall(pred_boxes, gt_boxes, iou_threshold)
            all_precisions.append(precision)
            all_recalls.append(recall)
        elif pred_boxes and not gt_boxes:
            all_precisions.append(0.0)
            all_recalls.append(0.0)
        elif not pred_boxes and gt_boxes:
            all_precisions.append(0.0)
            all_recalls.append(0.0)

        iterator.set_postfix({"samples": i + 1})

    return {
        "num_samples": len(all_precisions),
        "mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "mean_precision": float(np.mean(all_precisions)) if all_precisions else 0.0,
        "mean_recall": float(np.mean(all_recalls)) if all_recalls else 0.0,
    }


def benchmark_on_jsonl(
    model,
    tokenizer,
    jsonl_path: str,
    image_dir: str,
    max_samples: Optional[int] = None,
    batch_size: int = 8,
) -> Dict[str, float]:
    from .dataset import DetectionDataset
    ds = DetectionDataset(
        data_path=jsonl_path,
        image_dir=image_dir,
        tokenizer=tokenizer,
    )
    return run_benchmark(model, tokenizer, ds, image_dir, max_samples=max_samples, batch_size=batch_size)

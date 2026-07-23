import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
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


def compute_category_ap(
    pred_labels_boxes: Dict[int, List[Tuple[str, List[float]]]],
    gt_labels_boxes: Dict[int, List[Tuple[str, List[float]]]],
    iou_threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Compute per-category AP using IoU matching.
    
    Args:
        pred_labels_boxes: {image_id: [(label, [x1,y1,x2,y2]), ...]}
        gt_labels_boxes: {image_id: [(label, [x1,y1,x2,y2]), ...]}
    
    Returns:
        {category: {"AP": float, "precision": float, "recall": float, "support": int}}
    """
    categories = set()
    for img_id, items in gt_labels_boxes.items():
        for label, _ in items:
            categories.add(label)
    for img_id, items in pred_labels_boxes.items():
        for label, _ in items:
            categories.add(label)

    results = {}
    for cat in sorted(categories):
        cat_pred_by_image = {}
        cat_gt_by_image = {}
        for img_id in gt_labels_boxes:
            cat_gts = [(l, b) for l, b in gt_labels_boxes[img_id] if l == cat]
            if cat_gts:
                cat_gt_by_image[img_id] = cat_gts
            cat_preds = [(l, b) for l, b in pred_labels_boxes.get(img_id, []) if l == cat]
            if cat_preds:
                cat_pred_by_image[img_id] = cat_preds

        pred_boxes_flat = {img_id: [b for _, b in preds] for img_id, preds in cat_pred_by_image.items()}
        gt_boxes_flat = {img_id: [b for _, b in gts] for img_id, gts in cat_gt_by_image.items()}

        n_gt = sum(len(gts) for gts in cat_gt_by_image.values())
        ap = compute_ap(pred_boxes_flat, gt_boxes_flat, iou_threshold)

        precs, recs = [], []
        for img_id in cat_gt_by_image:
            preds = [b for _, b in cat_pred_by_image.get(img_id, [])]
            gts = [b for _, b in cat_gt_by_image[img_id]]
            p, r, _ = compute_precision_recall(preds, gts, iou_threshold)
            precs.append(p)
            recs.append(r)

        results[cat] = {
            "AP": ap,
            "precision": float(np.mean(precs)) if precs else 0.0,
            "recall": float(np.mean(recs)) if recs else 0.0,
            "support": n_gt,
        }
    return results


def visualize_eval_result(
    image: Image.Image,
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    gt_labels: Optional[List[str]] = None,
    pred_labels: Optional[List[str]] = None,
    prompt: str = "",
    output_path: Optional[str] = None,
) -> Image.Image:
    """Draw GT boxes (green) and predicted boxes (red) side-by-side on the image.
    
    GT boxes: green solid outline, labels in green.
    Pred boxes: red dashed-style outline, labels in red.
    Each pair of boxes is annotated with IoU if matched.
    """
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    for i, box in enumerate(gt_boxes):
        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline="#2ecc71", width=3)
        label = (gt_labels[i] if gt_labels and i < len(gt_labels) else None) or f"GT{i}"
        bbox = draw.textbbox((0, 0), label, font=font_small)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - th - 4, x1 + tw + 4, y1], fill="#2ecc71")
        draw.text((x1 + 2, y1 - th - 2), label, fill="white", font=font_small)

    for i, box in enumerate(pred_boxes):
        x1, y1, x2, y2 = map(int, box)
        for offset in range(2):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline="#e74c3c", width=2)
        label = (pred_labels[i] if pred_labels and i < len(pred_labels) else None) or f"P{i}"
        best_iou = 0.0
        for gt in gt_boxes:
            iou_val = _compute_iou_flat(box, gt)
            if iou_val > best_iou:
                best_iou = iou_val
        suffix = f" IoU={best_iou:.2f}" if best_iou > 0 else ""
        full_label = f"{label}{suffix}"
        bbox = draw.textbbox((0, 0), full_label, font=font_small)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        y_pos = y2 + 2
        draw.rectangle([x1, y_pos, x1 + tw + 4, y_pos + th + 4], fill="#e74c3c")
        draw.text((x1 + 2, y_pos + 2), full_label, fill="white", font=font_small)

    if prompt:
        clean = prompt.replace("<|image|>\n", "").replace("<|image|>", "")
        bbox = draw.textbbox((0, 0), clean, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([0, 0, tw + 8, th + 8], fill="black")
        draw.text((4, 4), clean, fill="white", font=font)

    legend_y = vis.height - 30
    draw.rectangle([0, legend_y, vis.width, vis.height], fill="black")
    draw.text((4, legend_y + 6), "GT:", fill="#2ecc71", font=font_small)
    draw.text((34, legend_y + 6), "Pred:", fill="#e74c3c", font=font_small)

    if output_path:
        vis.save(output_path)
    return vis


def _compute_iou_flat(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _resolve_image_path(image_path: str, image_dir: str) -> Optional[str]:
    resolved = image_path if os.path.isabs(image_path) else os.path.join(image_dir, image_path)
    if not os.path.exists(resolved):
        resolved = os.path.join(image_dir, os.path.basename(image_path))
    return resolved if os.path.exists(resolved) else None


def _extract_human_prompt(raw: dict) -> str:
    """Extract the human/user prompt from a raw data dict."""
    if "conversations" in raw:
        for conv in raw["conversations"]:
            if conv.get("from") in ("human", "user"):
                return conv["value"]
    if "messages" in raw:
        for msg in raw["messages"]:
            if msg.get("role") == "user":
                return msg.get("content", "")
    return ""


def _extract_gt_text(raw: dict) -> Optional[str]:
    """Extract the assistant/GPT response from a raw data dict."""
    if "conversations" in raw:
        for conv in raw["conversations"]:
            if conv.get("from") in ("gpt", "assistant"):
                return conv["value"]
    if "messages" in raw:
        for msg in raw["messages"]:
            if msg.get("role") == "assistant":
                return msg.get("content", "")
    return None


def run_benchmark(
    model,
    tokenizer,
    dataset,
    image_dir: str,
    max_samples: Optional[int] = None,
    iou_threshold: float = 0.5,
    batch_size: int = 8,
    save_vis_dir: Optional[str] = None,
    max_vis_images: int = 100,
    log_json: Optional[str] = None,
) -> Dict[str, float]:
    from .config import InferenceConfig
    from .inference import DetectionInferenceEngine
    from .utils import parse_boxes_from_text, parse_labels_and_boxes

    inf_cfg = InferenceConfig(max_new_tokens=512)
    engine = DetectionInferenceEngine(model, tokenizer, inf_cfg)

    if save_vis_dir:
        os.makedirs(save_vis_dir, exist_ok=True)

    pred_boxes_by_image = {}
    gt_boxes_by_image = {}
    pred_labels_boxes_by_image = {}
    gt_labels_boxes_by_image = {}
    all_ious = []
    all_precisions = []
    all_recalls = []
    per_image_log = []
    vis_count = 0

    from PIL import Image

    valid_indices = []
    for i in range(len(dataset)):
        if max_samples and i >= max_samples:
            break
        raw = dataset._raw_data[i]
        image_path = raw.get("image", "") or raw.get("image_path", "")
        resolved = _resolve_image_path(image_path, image_dir)
        if resolved is None:
            continue
        gt_text = _extract_gt_text(raw)
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
        batch_gt_labels_boxes = []
        batch_prompts = []
        batch_ids = []
        batch_image_names = []

        for idx in batch_indices:
            raw = dataset._raw_data[idx]
            image_path = raw.get("image", "") or raw.get("image_path", "")
            resolved = _resolve_image_path(image_path, image_dir)
            image = Image.open(resolved).convert("RGB")
            batch_images.append(image)
            batch_ids.append(idx)
            batch_image_names.append(os.path.basename(image_path))

            gt_text = _extract_gt_text(raw)
            gt_boxes = parse_boxes_from_text(gt_text or "")
            batch_gt_boxes.append(gt_boxes)
            gt_labeled = parse_labels_and_boxes(gt_text or "")
            batch_gt_labels_boxes.append(gt_labeled)

            human_text = _extract_human_prompt(raw)
            prompt = _make_prompt(human_text)
            batch_prompts.append(prompt)

        batch_results = engine.predict_batch(
            batch_images, [batch_prompts[0]] * len(batch_images), batch_size=len(batch_images)
        )

        for j, result in enumerate(batch_results):
            pred_boxes = result["boxes"]
            gt_boxes = batch_gt_boxes[j]
            img_id = batch_ids[j]
            orig_w, orig_h = batch_images[j].size

            pred_boxes_by_image[img_id] = pred_boxes
            gt_boxes_by_image[img_id] = gt_boxes

            pred_labeled = parse_labels_and_boxes(result.get("text", ""))
            pred_labels_boxes_by_image[img_id] = [(l, [
                int(b[0] * orig_w / 1000),
                int(b[1] * orig_h / 1000),
                int(b[2] * orig_w / 1000),
                int(b[3] * orig_h / 1000),
            ]) for l, b in pred_labeled]

            gt_labeled_scaled = [(l, [
                b[0] * orig_w / 1000,
                b[1] * orig_h / 1000,
                b[2] * orig_w / 1000,
                b[3] * orig_h / 1000,
            ]) for l, b in batch_gt_labels_boxes[j]]
            gt_labels_boxes_by_image[img_id] = gt_labeled_scaled

            best_iou = 0.0
            if pred_boxes and gt_boxes:
                box_ious = [compute_iou(p, g) for p in pred_boxes for g in gt_boxes]
                best_iou = max(box_ious)
                all_ious.append(best_iou)
                p, r, _ = compute_precision_recall(pred_boxes, gt_boxes, iou_threshold)
                all_precisions.append(p)
                all_recalls.append(r)
            elif not pred_boxes and gt_boxes:
                all_precisions.append(0.0)
                all_recalls.append(0.0)
            elif pred_boxes and not gt_boxes:
                all_precisions.append(0.0)
                all_recalls.append(0.0)

            entry = {
                "image": batch_image_names[j],
                "prompt": batch_prompts[j].replace("<|image|>\n", ""),
                "gt_text": _extract_gt_text(dataset._raw_data[img_id]) or "",
                "pred_text": result.get("text", ""),
                "gt_boxes": [[int(c) for c in b] for b in gt_boxes],
                "pred_boxes": [[int(c) for c in b] for b in pred_boxes],
                "gt_labels_boxes": [[l, [int(c) for c in b]] for l, b in gt_labeled_scaled],
                "pred_labels_boxes": [[l, [int(c) for c in b]] for l, b in pred_labels_boxes_by_image[img_id]],
                "best_iou": best_iou,
            }
            per_image_log.append(entry)

            if save_vis_dir and vis_count < max_vis_images:
                vis_name = os.path.splitext(batch_image_names[j])[0] + ".jpg"
                vis_path = os.path.join(save_vis_dir, vis_name)
                visualize_eval_result(
                    batch_images[j].copy(),
                    [b for _, b in gt_labeled_scaled],
                    [b for _, b in pred_labels_boxes_by_image[img_id]],
                    gt_labels=[l for l, _ in gt_labeled_scaled],
                    pred_labels=[l for l, _ in pred_labels_boxes_by_image[img_id]],
                    prompt=batch_prompts[j],
                    output_path=vis_path,
                )
                vis_count += 1

        iterator.set_postfix({"samples": min(end_idx, num_samples)})

    if log_json:
        os.makedirs(os.path.dirname(log_json) or ".", exist_ok=True)
        with open(log_json, "w") as f:
            json.dump(per_image_log, f, indent=2)

    results = {
        "num_samples": len(all_precisions),
        "mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
    }

    coco_aps = compute_coco_ap(pred_boxes_by_image, gt_boxes_by_image)
    results.update(coco_aps)

    cat_aps = compute_category_ap(pred_labels_boxes_by_image, gt_labels_boxes_by_image)
    if cat_aps:
        cat_ap_vals = [v["AP"] for v in cat_aps.values()]
        results["mAP_per_category"] = float(np.mean(cat_ap_vals))
        results["per_category"] = {
            cat: {"AP": v["AP"], "support": v["support"]}
            for cat, v in sorted(cat_aps.items(), key=lambda x: -x[1]["AP"])
        }

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
    save_vis_dir: Optional[str] = None,
    max_vis_images: int = 100,
    log_json: Optional[str] = None,
) -> Dict[str, float]:
    from .dataset import DetectionDataset
    ds = DetectionDataset(
        data_path=jsonl_path,
        image_dir=image_dir,
        tokenizer=tokenizer,
    )
    return run_benchmark(model, tokenizer, ds, image_dir, max_samples=max_samples,
                         batch_size=batch_size, save_vis_dir=save_vis_dir,
                         max_vis_images=max_vis_images, log_json=log_json)

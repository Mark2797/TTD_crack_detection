#!/usr/bin/env python3
"""Validation-tuned connected-component post-processing for crack segmentation.

This script avoids test-set leakage:
1. Run the trained model on the validation split.
2. Sweep a list of minimum connected-component areas.
3. Select the area threshold using validation mean F1.
4. Evaluate the original and selected post-processed masks on the held-out test split.

The intended use is a small post-processing baseline for the ResNet segmentation
model, not a replacement for the original training/evaluation script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "baseline_models_scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from train_ttd import EXPERIMENTS, baseline_model_pipeline, build_model  # noqa: E402


ARCH_OPTIONS = [
    "DeepLabV3Plus",
    "DeepLabV3",
    "UnetPlusPlus",
    "Segformer",
    "Unet",
    "FPN",
    "PSPNet",
    "Linknet",
    "PAN",
    "MAnet",
]


@dataclass(frozen=True)
class ModelConfig:
    arch: str
    encoder: str
    weights: str
    setting: str


def parse_model_name(model_name: str) -> ModelConfig:
    if "_" not in model_name:
        raise ValueError(f"Model name must include setting after '_': {model_name}")
    model_part, setting = model_name.split("_", 1)
    for arch in sorted(ARCH_OPTIONS, key=len, reverse=True):
        prefix = f"{arch}-"
        if model_part.startswith(prefix):
            rest = model_part[len(prefix) :]
            parts = rest.split("-")
            if len(parts) < 2:
                raise ValueError(f"Cannot parse encoder/weights from: {model_name}")
            return ModelConfig(
                arch=arch,
                encoder="-".join(parts[:-1]),
                weights=parts[-1],
                setting=setting,
            )
    raise ValueError(f"Unsupported architecture in model name: {model_name}")


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove foreground connected components smaller than min_area pixels."""
    mask_bool = mask.astype(bool)
    if min_area <= 0 or not mask_bool.any():
        return mask_bool.astype(np.uint8)

    labels, n_labels = ndimage.label(mask_bool)
    if n_labels == 0:
        return mask_bool.astype(np.uint8)

    counts = np.bincount(labels.ravel())
    keep_labels = np.where(counts >= min_area)[0]
    keep_labels = keep_labels[keep_labels != 0]
    if len(keep_labels) == 0:
        return np.zeros_like(mask_bool, dtype=np.uint8)
    return np.isin(labels, keep_labels).astype(np.uint8)


def confusion_counts(pred: np.ndarray, target: np.ndarray) -> tuple[int, int, int]:
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)
    tp = int((pred_bool & target_bool).sum())
    fp = int((pred_bool & ~target_bool).sum())
    fn = int((~pred_bool & target_bool).sum())
    return tp, fp, fn


def metrics_from_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    iou_den = tp + fp + fn
    precision_den = tp + fp
    recall_den = tp + fn

    iou = 1.0 if iou_den == 0 else tp / iou_den
    precision = 1.0 if precision_den == 0 else tp / precision_den
    recall = 1.0 if recall_den == 0 else tp / recall_den
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def aggregate_per_image(rows: list[dict[str, float | int | str]]) -> dict[str, float | int]:
    tp = int(sum(int(r["tp_pixels"]) for r in rows))
    fp = int(sum(int(r["fp_pixels"]) for r in rows))
    fn = int(sum(int(r["fn_pixels"]) for r in rows))
    micro = metrics_from_counts(tp, fp, fn)
    return {
        "n_images": len(rows),
        "mean_iou": float(np.mean([float(r["iou"]) for r in rows])) if rows else math.nan,
        "mean_f1": float(np.mean([float(r["f1"]) for r in rows])) if rows else math.nan,
        "mean_precision": float(np.mean([float(r["precision"]) for r in rows])) if rows else math.nan,
        "mean_recall": float(np.mean([float(r["recall"]) for r in rows])) if rows else math.nan,
        "micro_iou": micro["iou"],
        "micro_f1": micro["f1"],
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
        "empty_gt_with_prediction": int(
            sum(int(r["gt_pixels"]) == 0 and int(r["pred_pixels"]) > 0 for r in rows)
        ),
        "positive_gt_empty_prediction": int(
            sum(int(r["gt_pixels"]) > 0 and int(r["pred_pixels"]) == 0 for r in rows)
        ),
    }


def predict_loader(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    threshold: float,
) -> list[dict[str, np.ndarray | int]]:
    """Return probability, raw prediction, and target arrays for each image."""
    model.eval()
    records: list[dict[str, np.ndarray | int]] = []
    sample_id = 0
    with torch.no_grad():
        for images, masks in dataloader:
            images = torch.as_tensor(images).to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            preds = (probs >= threshold).astype(np.uint8)
            targets = masks.squeeze(1).cpu().numpy() if masks.ndim == 4 else masks.cpu().numpy()
            for i in range(images.shape[0]):
                records.append(
                    {
                        "sample_id": sample_id,
                        "prob": probs[i],
                        "raw_pred": preds[i],
                        "target": targets[i].astype(np.uint8),
                    }
                )
                sample_id += 1
    return records


def evaluate_records(
    records: list[dict[str, np.ndarray | int]],
    min_area: int,
    split_name: str,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    rows: list[dict[str, float | int | str]] = []
    for item in records:
        raw_pred = item["raw_pred"]
        target = item["target"]
        assert isinstance(raw_pred, np.ndarray)
        assert isinstance(target, np.ndarray)
        pred = remove_small_components(raw_pred, min_area=min_area)
        tp, fp, fn = confusion_counts(pred, target)
        metrics = metrics_from_counts(tp, fp, fn)
        rows.append(
            {
                "split": split_name,
                "sample_id": int(item["sample_id"]),
                "min_component_area": int(min_area),
                "gt_pixels": int(target.astype(bool).sum()),
                "raw_pred_pixels": int(raw_pred.astype(bool).sum()),
                "pred_pixels": int(pred.astype(bool).sum()),
                "tp_pixels": tp,
                "fp_pixels": fp,
                "fn_pixels": fn,
                **metrics,
            }
        )

    summary = aggregate_per_image(rows)
    summary.update({"split": split_name, "min_component_area": int(min_area)})
    return summary, rows


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    runs_dir = project_root / "baseline_models_scripts" / "runs"
    models_dir = runs_dir / "models"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = parse_model_name(args.model_name)
    if args.setting and args.setting != cfg.setting:
        raise ValueError(f"--setting {args.setting} does not match model setting {cfg.setting}")
    setting = cfg.setting

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"device = {device}")
    print(f"model = {args.model_name}")
    print(f"setting = {setting}")

    train_dl, val_dl, test_dl, _custom_stats, _df_test_ready = baseline_model_pipeline(
        base_dir=project_root,
        dict_files=EXPERIMENTS[setting],
        bs=args.batch_size,
        img_size=args.img_size,
    )
    del train_dl

    model = build_model(cfg.arch, cfg.encoder, cfg.weights, classes=2)
    ckpt_path = models_dir / f"{args.model_name}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    min_areas = [int(x) for x in args.min_areas.split(",") if x.strip()]
    if 0 not in min_areas:
        min_areas = [0] + min_areas

    print("Predicting validation split...")
    val_records = predict_loader(model, val_dl, device=device, threshold=args.threshold)
    print("Predicting test split...")
    test_records = predict_loader(model, test_dl, device=device, threshold=args.threshold)

    val_summaries = []
    for min_area in min_areas:
        summary, _rows = evaluate_records(val_records, min_area=min_area, split_name="val")
        val_summaries.append(summary)

    val_summaries = sorted(
        val_summaries,
        key=lambda r: (
            float(r[args.selection_metric]),
            float(r["mean_iou"]),
            -int(r["min_component_area"]),
        ),
        reverse=True,
    )
    best = val_summaries[0]
    best_min_area = int(best["min_component_area"])

    raw_test_summary, raw_test_rows = evaluate_records(test_records, min_area=0, split_name="test")
    best_test_summary, best_test_rows = evaluate_records(
        test_records,
        min_area=best_min_area,
        split_name="test",
    )

    result_prefix = output_dir / f"connected_components_{args.model_name}_thr{str(args.threshold).replace('.', 'p')}"
    write_csv(result_prefix.with_name(result_prefix.name + "_val_sweep.csv"), val_summaries)
    write_csv(
        result_prefix.with_name(result_prefix.name + "_test_summary.csv"),
        [
            {"postprocess": "none", **raw_test_summary},
            {"postprocess": "remove_small_components", **best_test_summary},
        ],
    )
    write_csv(result_prefix.with_name(result_prefix.name + "_test_per_image_raw.csv"), raw_test_rows)
    write_csv(result_prefix.with_name(result_prefix.name + "_test_per_image_selected.csv"), best_test_rows)

    metadata = {
        "model_name": args.model_name,
        "setting": setting,
        "threshold": args.threshold,
        "min_areas": min_areas,
        "selection_metric": args.selection_metric,
        "selected_min_component_area": best_min_area,
        "selected_validation_summary": best,
        "raw_test_summary": raw_test_summary,
        "selected_test_summary": best_test_summary,
        "note": "min_component_area selected on validation split only; test split evaluated after selection.",
    }
    result_prefix.with_name(result_prefix.name + "_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("\nValidation-selected post-processing")
    print(f"selected min_component_area = {best_min_area}")
    print(f"validation {args.selection_metric} = {float(best[args.selection_metric]):.4f}")
    print("\nTest summary")
    print(pd.DataFrame([{"postprocess": "none", **raw_test_summary}, {"postprocess": "remove_small_components", **best_test_summary}]))
    print(f"\nSaved outputs with prefix: {result_prefix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--model-name", default="Unet-resnet34-imagenet_Single-TB")
    parser.add_argument("--setting", default=None)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--min-areas", default="0,5,10,20,30,50,75,100,150,200,300,500")
    parser.add_argument("--selection-metric", default="mean_f1", choices=["mean_f1", "mean_iou", "micro_f1", "micro_iou"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "explainability_suite" / "outputs" / "postprocessing",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

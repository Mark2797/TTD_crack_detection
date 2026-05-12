from __future__ import annotations

from typing import Dict

import numpy as np


def _binarize(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def confusion_stats(pred_mask: np.ndarray, true_mask: np.ndarray) -> Dict[str, float]:
    pred = _binarize(pred_mask)
    true = _binarize(true_mask)

    tp = float(((pred == 1) & (true == 1)).sum())
    fp = float(((pred == 1) & (true == 0)).sum())
    fn = float(((pred == 0) & (true == 1)).sum())
    tn = float(((pred == 0) & (true == 0)).sum())

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def iou(pred_mask: np.ndarray, true_mask: np.ndarray, smooth: float = 1e-6) -> float:
    stats = confusion_stats(pred_mask, true_mask)
    return (stats["tp"] + smooth) / (stats["tp"] + stats["fp"] + stats["fn"] + smooth)


def precision(pred_mask: np.ndarray, true_mask: np.ndarray, smooth: float = 1e-6) -> float:
    stats = confusion_stats(pred_mask, true_mask)
    return (stats["tp"] + smooth) / (stats["tp"] + stats["fp"] + smooth)


def recall(pred_mask: np.ndarray, true_mask: np.ndarray, smooth: float = 1e-6) -> float:
    stats = confusion_stats(pred_mask, true_mask)
    return (stats["tp"] + smooth) / (stats["tp"] + stats["fn"] + smooth)


def f1(pred_mask: np.ndarray, true_mask: np.ndarray, smooth: float = 1e-6) -> float:
    p = precision(pred_mask, true_mask, smooth=smooth)
    r = recall(pred_mask, true_mask, smooth=smooth)
    return 2.0 * (p * r) / (p + r + smooth)

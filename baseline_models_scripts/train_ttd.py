#!/usr/bin/env python3
"""Train TACK Tunnel Data segmentation models from a notebook-derived pipeline.

This script is adapted from the uploaded baseline notebook and adds CLI arguments
so you can switch architecture / encoder(backbone) / experiment without editing
cells by hand.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Dict, List, Tuple

# Make the project root importable when this file is placed in baseline_models/
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parent
if str(DEFAULT_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(DEFAULT_PROJECT_ROOT))

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from fastai.losses import CrossEntropyLossFlat, DiceLoss

try:
    import segmentation_models_pytorch as smp
except ModuleNotFoundError as exc:
    raise SystemExit(
        "segmentation_models_pytorch is not installed in this environment.\n"
        "Install it in your venv first, e.g.\n"
        "  python -m pip install segmentation-models-pytorch\n"
    ) from exc

from matrices import * 
from preprocessing import * 
from utils import * 


EXPERIMENTS: Dict[str, Dict[str, List[str]]] = {
    "Single-TA": {
        "train_files": ["TA_train.csv"],
        "val_files": ["TA_val.csv"],
        "test_files": ["TA_test.csv"],
    },
    "Single-TB": {
        "train_files": ["TB_train.csv"],
        "val_files": ["TB_val.csv"],
        "test_files": ["TB_test.csv"],
    },
    "Single-TC": {
        "train_files": ["TC_train.csv"],
        "val_files": ["TC_val.csv"],
        "test_files": ["TC_test.csv"],
    },
    "Multi-Domain": {
        "train_files": ["TA_train.csv", "TB_train.csv", "TC_train.csv"],
        "val_files": ["TA_val.csv", "TB_val.csv", "TC_val.csv"],
        "test_files": ["TA_test.csv", "TB_test.csv", "TC_test.csv"],
    },
    "Shift-TA_TB-to-TC_10pct": {
        "train_files": ["TA_train.csv", "TB_train.csv"],
        "val_files": ["TA_val.csv", "TB_val.csv"],
        "test_files": ["TC_test.csv"],
    },
    "Shift-TA_TB-to-TC_100pct": {
        "train_files": ["TA_train.csv", "TB_train.csv"],
        "val_files": ["TA_val.csv", "TB_val.csv"],
        "test_files": ["TC_train.csv", "TC_val.csv", "TC_test.csv"],
    },
    "Shift-TA_TC-to-TB_10pct": {
        "train_files": ["TA_train.csv", "TC_train.csv"],
        "val_files": ["TA_val.csv", "TC_val.csv"],
        "test_files": ["TB_test.csv"],
    },
    "Shift-TA_TC-to-TB_100pct": {
        "train_files": ["TA_train.csv", "TC_train.csv"],
        "val_files": ["TA_val.csv", "TC_val.csv"],
        "test_files": ["TB_train.csv", "TB_val.csv", "TB_test.csv"],
    },
    "Shift-TB_TC-to-TA_10pct": {
        "train_files": ["TB_train.csv", "TC_train.csv"],
        "val_files": ["TB_val.csv", "TC_val.csv"],
        "test_files": ["TA_test.csv"],
    },
    "Shift-TB_TC-to-TA_100pct": {
        "train_files": ["TB_train.csv", "TC_train.csv"],
        "val_files": ["TB_val.csv", "TC_val.csv"],
        "test_files": ["TA_train.csv", "TA_val.csv", "TA_test.csv"],
    },
}


class TTDCombinedLoss(nn.Module):
    def __init__(self, ce_weight_tensor: torch.Tensor, w_ce: float = 0.5, w_dice: float = 0.5):
        super().__init__()
        self.w_ce = w_ce
        self.w_dice = w_dice
        self.ce_loss = CrossEntropyLossFlat(weight=ce_weight_tensor, axis=1)
        self.dice_loss = DiceLoss(axis=1)

    def forward(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        pred_tensor = pred.as_subclass(torch.Tensor)
        targ_tensor = targ.as_subclass(torch.Tensor).long()
        ce = self.ce_loss(pred_tensor, targ_tensor)
        dice = self.dice_loss(pred_tensor, targ_tensor)
        return self.w_ce * ce + self.w_dice * dice


def baseline_model_pipeline(base_dir: Path, dict_files: Dict[str, List[str]], bs: int = 16, img_size: int = 512):
    dataset_folder = base_dir / "TACK_Tunnel_Data"
    csv_source_dir = dataset_folder / "2_model_input"
    raw_mask_dir = dataset_folder / "3_mask"

    pipeline = TunnelDataPipeline(base_dir=str(dataset_folder), original_mask_dir=str(raw_mask_dir))

    print("Loading CSV metadata...")
    df_train_val, df_test = pipeline.load_csv_data(
        csv_source_dir=str(csv_source_dir),
        train_files=dict_files["train_files"],
        val_files=dict_files["val_files"],
        test_files=dict_files["test_files"],
    )

    print("Sanitizing training and validation masks...")
    df_train_val_ready = pipeline.sanitize_masks(df_train_val, class_pixel_value=40)

    print("Sanitizing test masks...")
    df_test_ready = pipeline.sanitize_masks(df_test, class_pixel_value=40)

    custom_stats = None  # keep the notebook baseline behavior

    print("Generating Dataloaders...")
    train_dl, val_dl, test_dl = pipeline.get_dataloaders(
        train_val_df=df_train_val_ready,
        test_df=df_test_ready,
        bs=bs,
        img_size=img_size,
        custom_stats=custom_stats,
    )

    print("\nPipeline Ready:")
    print(f" - Training batches: {len(train_dl)}")
    print(f" - Validation batches: {len(val_dl)}")
    print(f" - Testing batches: {len(test_dl)}")

    return train_dl, val_dl, test_dl, custom_stats, df_test_ready


def build_model(arch: str, encoder_name: str, encoder_weights: str | None, classes: int):
    arch_map = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "fpn": smp.FPN,
        "pspnet": smp.PSPNet,
        "linknet": smp.Linknet,
        "pan": smp.PAN,
        "manet": smp.MAnet,
        "deeplabv3": smp.DeepLabV3,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "segformer": smp.Segformer,
    }
    arch_key = arch.lower()
    if arch_key not in arch_map:
        raise ValueError(f"Unsupported arch '{arch}'. Choose from: {', '.join(arch_map)}")
    return arch_map[arch_key](encoder_name=encoder_name, encoder_weights=encoder_weights, classes=classes)


def safe_name(text: str) -> str:
    return text.replace("/", "-").replace(" ", "_")


def run_one_experiment(
    exp_name: str,
    exp_cfg: Dict[str, List[str]],
    args: argparse.Namespace,
    device: torch.device,
    models_dir: Path,
    figures_dir: Path,
) -> Dict[str, float | str]:
    if args.quiet:
        context = redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())
    else:
        context = None

    def _inner():
        train_dl, val_dl, test_dl, custom_stats, df_test_ready = baseline_model_pipeline(
            base_dir=args.project_root,
            dict_files=exp_cfg,
            bs=args.batch_size,
            img_size=args.img_size,
        )

        model = build_model(
            arch=args.arch,
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weights,
            classes=args.classes,
        )
        model_name = safe_name(f"{args.arch}-{args.encoder}-{args.encoder_weights}_{exp_name}")

        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.eta_min,
        )

        weights = torch.tensor([1.0, args.crack_weight], device=device)
        loss_fn = TTDCombinedLoss(ce_weight_tensor=weights, w_ce=args.w_ce, w_dice=args.w_dice)

        history = epochs(
            model,
            model_name,
            device,
            train_dl,
            val_dl,
            loss_fn,
            optimizer,
            args.epochs,
            scheduler=scheduler,
            patience=args.patience,
            save_dir=str(models_dir),
        )
        save_training_history(history, model_name, save_dir=str(figures_dir))

        ckpt_path = models_dir / f"{model_name}.pth"
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)

        v_loss, v_iou, v_f1, v_recall, v_prec = val_loop(model, device, val_dl, loss_fn, is_test=True)
        t_loss, t_iou, t_f1, t_recall, t_prec = val_loop(model, device, test_dl, loss_fn, is_test=True)

        save_prediction_overlap(model, model_name, test_dl, device, custom_stats=custom_stats)

        return {
            "Experiment": model_name,
            "Val_Loss": v_loss,
            "Val_IoU": v_iou,
            "Val_F1": v_f1,
            "Val_Recall": v_recall,
            "Val_Prec": v_prec,
            "Test_Loss": t_loss,
            "Test_IoU": t_iou,
            "Test_F1": t_f1,
            "Test_Recall": t_recall,
            "Test_Prec": t_prec,
        }

    if context is None:
        return _inner()

    with context[0], context[1]:
        return _inner()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TACK tunnel baseline from notebook logic.")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT, help="Project root containing TACK_Tunnel_Data")
    parser.add_argument("--experiment", default="all", help="Experiment name, comma-separated names, or 'all'")
    parser.add_argument("--arch", default="Unet", help="smp architecture, e.g. Unet, FPN, DeepLabV3Plus")
    parser.add_argument("--encoder", default="resnet34", help="Backbone / encoder_name for segmentation_models_pytorch")
    parser.add_argument("--encoder-weights", default="imagenet", help="Encoder weights, usually 'imagenet' or None")
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eta-min", type=float, default=1e-7)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--crack-weight", type=float, default=20.0)
    parser.add_argument("--w-ce", type=float, default=0.5)
    parser.add_argument("--w-dice", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "runs")
    parser.add_argument("--results-name", default=None, help="Optional CSV filename")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose training / pipeline stdout")
    return parser.parse_args()


def resolve_experiments(raw: str) -> List[Tuple[str, Dict[str, List[str]]]]:
    if raw.lower() == "all":
        return list(EXPERIMENTS.items())

    names = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [name for name in names if name not in EXPERIMENTS]
    if unknown:
        raise SystemExit(
            f"Unknown experiment(s): {unknown}\nAvailable: {', '.join(EXPERIMENTS.keys())}"
        )
    return [(name, EXPERIMENTS[name]) for name in names]


def main() -> None:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.output_dir = args.output_dir.resolve()

    models_dir = args.output_dir / "models"
    figures_dir = args.output_dir / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Python executable: {sys.executable}")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            print(f"GPU {idx}: {torch.cuda.get_device_name(idx)}")
    print(f"Using device: {device}")

    selected_experiments = resolve_experiments(args.experiment)
    rows = []
    for exp_name, exp_cfg in selected_experiments:
        print(f"\n===== Running {exp_name} | arch={args.arch} | encoder={args.encoder} =====")
        row = run_one_experiment(exp_name, exp_cfg, args, device, models_dir, figures_dir)
        rows.append(row)

        results_csv = args.output_dir / (
            args.results_name
            if args.results_name
            else f"results_{safe_name(args.arch)}_{safe_name(args.encoder)}.csv"
        )
        # pd.DataFrame(rows).to_csv(results_csv, index=False)
        print(f"Saved interim results to: {results_csv}")

    final_df = pd.DataFrame(rows)
    print("\nFinal results:")
    print(final_df)


if __name__ == "__main__":
    main()

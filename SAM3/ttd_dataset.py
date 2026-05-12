from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image


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


@dataclass(frozen=True)
class TTDPaths:
    project_root: Path

    @property
    def dataset_root(self) -> Path:
        return self.project_root / "TACK_Tunnel_Data"

    @property
    def csv_dir(self) -> Path:
        return self.dataset_root / "2_model_input"

    @property
    def raw_mask_dir(self) -> Path:
        return self.dataset_root / "3_mask"

    @property
    def sanitized_mask_dir(self) -> Path:
        return self.dataset_root / "3_masks_sanitized"


class TTDDatasetBuilder:
    def __init__(self, project_root: str | Path):
        self.paths = TTDPaths(project_root=Path(project_root).resolve())
        self.paths.sanitized_mask_dir.mkdir(parents=True, exist_ok=True)

    def load_experiment(self, experiment_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if experiment_name not in EXPERIMENTS:
            raise ValueError(
                f"Unknown experiment '{experiment_name}'. "
                f"Available: {', '.join(EXPERIMENTS)}"
            )
        cfg = EXPERIMENTS[experiment_name]
        return self.load_csv_data(
            train_files=cfg["train_files"],
            val_files=cfg["val_files"],
            test_files=cfg["test_files"],
        )

    def load_csv_data(
        self,
        train_files: Sequence[str],
        val_files: Sequence[str],
        test_files: Optional[Sequence[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        def _read(files: Iterable[str], is_valid: bool, is_test: bool) -> pd.DataFrame:
            frames: List[pd.DataFrame] = []
            for filename in files:
                csv_path = self.paths.csv_dir / filename
                if not csv_path.exists():
                    continue
                df = pd.read_csv(csv_path)
                df["source_csv"] = filename
                df["is_valid"] = is_valid
                df["is_test"] = is_test
                frames.append(df)
            if not frames:
                return pd.DataFrame()
            return pd.concat(frames, ignore_index=True)

        train_val_df = pd.concat(
            [
                _read(train_files, is_valid=False, is_test=False),
                _read(val_files, is_valid=True, is_test=False),
            ],
            ignore_index=True,
        )
        test_df = (
            _read(test_files, is_valid=False, is_test=True)
            if test_files
            else pd.DataFrame()
        )
        return train_val_df, test_df

    def sanitize_masks(
        self,
        df: pd.DataFrame,
        class_pixel_value: int = 40,
        sanitized_value: int = 1,
    ) -> pd.DataFrame:
        image_abs_paths: List[str] = []
        sanitized_paths: List[str] = []
        valid_indices: List[int] = []

        for idx, row in df.iterrows():
            clean_filename = str(row["filename"]).split("../")[-1]
            abs_img_path = (self.paths.dataset_root / clean_filename).resolve()
            img_name = abs_img_path.stem
            mask_name = self._build_mask_filename(img_name)
            raw_mask_path = self.paths.raw_mask_dir / mask_name
            sanitized_mask_path = self.paths.sanitized_mask_dir / mask_name

            if not raw_mask_path.exists() or not abs_img_path.exists():
                continue

            if not sanitized_mask_path.exists():
                mask_arr = np.array(Image.open(raw_mask_path))
                new_mask = np.zeros_like(mask_arr, dtype=np.uint8)
                if int(row.get("target", 0)) == 1:
                    new_mask[mask_arr == class_pixel_value] = sanitized_value
                Image.fromarray(new_mask).save(sanitized_mask_path)

            image_abs_paths.append(str(abs_img_path))
            sanitized_paths.append(str(sanitized_mask_path))
            valid_indices.append(idx)

        out = df.iloc[valid_indices].copy()
        out["image_abs_path"] = image_abs_paths
        out["mask_path_sanitized"] = sanitized_paths
        return out

    def build_records(self, experiment_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_val_df, test_df = self.load_experiment(experiment_name)
        train_val_ready = self.sanitize_masks(train_val_df)
        test_ready = self.sanitize_masks(test_df)
        return train_val_ready, test_ready

    @staticmethod
    def _build_mask_filename(img_name: str) -> str:
        parts = img_name.rsplit("_", 1)
        if len(parts) == 2:
            return f"{parts[0]}_fuse_{parts[1]}_1band.png"
        return f"{img_name}.png"

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from fastai.vision.all import *

class TunnelDataPipeline:
    def __init__(self, base_dir, original_mask_dir, processed_img_subdir="4_img_sobelxy"):
        """
        Initialize the pipeline and establish the directory structure.
        Ensures output directories for sanitized masks exist outside the core dataset folder.
        """
        self.base_dir = base_dir or os.getcwd()
        self.original_mask_dir = original_mask_dir
        
        self.sanitized_mask_dir = os.path.join(self.base_dir, '3_masks_sanitized')
        self.processed_img_dir = os.path.join(self.base_dir, processed_img_subdir)
        
        os.makedirs(self.sanitized_mask_dir, exist_ok=True)

    def load_csv_data(self, csv_source_dir, train_files, val_files, test_files=None):
        """
        Load and combine CSV files for training, validation, and testing.
        Assigns an 'is_valid' flag for fastai splitting and 'is_test' for evaluation.
        """
        def process_files(files, is_valid=False, is_test=False):
            dfs = []
            for f in files:
                path = os.path.join(csv_source_dir, f)
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df['is_valid'] = is_valid
                    df['is_test'] = is_test
                    dfs.append(df)
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        df_train_val = pd.concat([
            process_files(train_files, is_valid=False),
            process_files(val_files, is_valid=True)
        ], ignore_index=True)

        df_test = process_files(test_files, is_test=True) if test_files else pd.DataFrame()

        return df_train_val, df_test

    def sanitize_masks(self, df, class_pixel_value=40, sanitized_value=1):
        """
        Convert multi-class masks into binary masks for a target class.
        Original TTD values: 40 (Crack), 160 (Water), 200 (Leaching).
        """
        image_abs_paths, image_processed_paths, sanitized_paths, valid_indices = [], [], [], []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Sanitizing Masks"):
            try:
                clean_filename = row['filename'].split('../')[-1]
                abs_img_path = os.path.normpath(os.path.join(self.base_dir, clean_filename))
                img_name = os.path.splitext(os.path.basename(abs_img_path))[0]

                # choose one suffix depending on the processed folder you are using
                if os.path.basename(self.processed_img_dir) == "4_img_sobelxy":
                    processed_img_name = f"{img_name}_sobelxy.png"
                elif os.path.basename(self.processed_img_dir) == "4_img_sobelmaglap":
                    processed_img_name = f"{img_name}_sobelmaglap.png"
                elif os.path.basename(self.processed_img_dir) == "4_img_sobelmag_gabor":
                    processed_img_name = f"{img_name}_sobelmag_gabor.png"
                else:
                    raise ValueError(f"Unknown processed image folder: {self.processed_img_dir}")

                processed_img_path = os.path.join(self.processed_img_dir, processed_img_name)

                parts = img_name.rsplit('_', 1)
                mask_fn = f"{parts[0]}_fuse_{parts[1]}_1band.png" if len(parts) == 2 else f"{img_name}.png"

                raw_path = os.path.join(self.original_mask_dir, mask_fn)
                clean_path = os.path.join(self.sanitized_mask_dir, mask_fn)

                if not os.path.exists(clean_path):
                    if not os.path.exists(raw_path):
                        print(f"Skipping: Raw mask not found at {raw_path}")
                        continue

                    mask_arr = np.array(Image.open(raw_path))
                    new_mask = np.zeros_like(mask_arr, dtype=np.uint8)

                    if int(row.get('target', 0)) == 1:
                        new_mask[mask_arr == class_pixel_value] = sanitized_value

                    Image.fromarray(new_mask).save(clean_path)

                if not os.path.exists(processed_img_path):
                    print(f"Skipping: Processed image not found at {processed_img_path}")
                    continue

                image_abs_paths.append(abs_img_path)
                image_processed_paths.append(processed_img_path)
                sanitized_paths.append(clean_path)
                valid_indices.append(idx)

            except Exception as e:
                print(f"Error processing {img_name}: {e}")

        df_clean = df.iloc[valid_indices].copy()
        df_clean['image_abs_path'] = image_abs_paths
        df_clean['image_processed_path'] = image_processed_paths
        df_clean['mask_path_sanitized'] = sanitized_paths
        return df_clean

    def calculate_training_stats(self, df):
        """
        Calculate mean and standard deviation across the training set for custom normalization.
        """
        print("Calculating custom dataset statistics...")
        train_df = df[(df['is_valid'] == False) & (df['is_test'] == False)]
        paths = train_df['image_processed_path'].values

        means, stds = [], []
        for p in tqdm(paths, desc="Computing Stats"):
            img = np.array(Image.open(p).convert('RGB')) / 255.0
            means.append(np.mean(img, axis=(0, 1)))
            stds.append(np.std(img, axis=(0, 1)))

        return torch.tensor(np.mean(means, axis=0)), torch.tensor(np.mean(stds, axis=0))

    def get_dataloaders(self, train_val_df, test_df=None, bs=16, img_size=512, custom_stats=None):
        """
        Build fastai DataLoaders.
        """
        codes = np.array(['background', 'defect'])
        norm_stats = custom_stats if custom_stats else imagenet_stats

        dblock = DataBlock(
            blocks=(ImageBlock, MaskBlock(codes)),
            get_x=ColReader('image_processed_path'),
            get_y=ColReader('mask_path_sanitized'),
            splitter=ColSplitter('is_valid'),
            item_tfms=Resize(img_size),
            batch_tfms=[
                *aug_transforms(flip_vert=True, max_rotate=15.0, max_zoom=1.1, max_lighting=0.2),
                Normalize.from_stats(*norm_stats)
            ]
        )

        dls = dblock.dataloaders(train_val_df, bs=bs, num_workers=0, pin_memory=False)

        test_dl = None
        if test_df is not None and not test_df.empty:
            test_dl = dls.test_dl(test_df, with_labels=True)

        return dls.train, dls.valid, test_dl


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_folder = os.path.join(base_dir, "TACK_Tunnel_Data")

    csv_source_dir = os.path.join(dataset_folder, "2_model_input")
    raw_mask_dir = os.path.join(dataset_folder, "3_mask")

    # choose one:
    processed_version = "4_img_sobelxy"
    # processed_version = "4_img_sobelmaglap"

    pipeline = TunnelDataPipeline(
        base_dir=dataset_folder,
        original_mask_dir=raw_mask_dir,
        processed_img_subdir=processed_version
    )

    train_files = ["TA_train.csv", "TB_train.csv", "TC_train.csv"]
    val_files = ["TA_val.csv", "TB_val.csv", "TC_val.csv"]
    test_files = ["TA_test.csv", "TB_test.csv", "TC_test.csv"]

    print("Loading CSV metadata...")
    df_train_val, df_test = pipeline.load_csv_data(
        csv_source_dir=csv_source_dir,
        train_files=train_files,
        val_files=val_files,
        test_files=test_files
    )

    print("Sanitizing training and validation masks...")
    df_train_val_ready = pipeline.sanitize_masks(df_train_val, class_pixel_value=40)

    print("Sanitizing test masks...")
    df_test_ready = pipeline.sanitize_masks(df_test, class_pixel_value=40)

    custom_stats = pipeline.calculate_training_stats(df_train_val_ready)

    print("Generating Dataloaders...")
    train_dl, val_dl, test_dl = pipeline.get_dataloaders(
        train_val_df=df_train_val_ready,
        test_df=df_test_ready,
        bs=16,
        img_size=512,
        custom_stats=custom_stats
    )

    print(f"\nPipeline Ready:")
    print(f" - Training batches: {len(train_dl)}")
    print(f" - Validation batches: {len(val_dl)}")
    print(f" - Testing batches: {len(test_dl)}")
    print(custom_stats)

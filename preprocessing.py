import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from fastai.vision.all import *

class TunnelDataPipeline:
    def __init__(self, base_dir, original_mask_dir):
        """
        Initialize the pipeline and establish the directory structure.
        """
        self.base_dir = base_dir or os.getcwd()
        self.original_mask_dir = original_mask_dir
        
        # Internal directories for sanitized masks and model outputs
        self.sanitized_mask_dir = os.path.join(self.base_dir, '3_masks_sanitized')
        
        # Create directories if they do not exist
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
                    df['is_valid'] = is_valid # Used by fastai ColSplitter
                    df['is_test'] = is_test
                    dfs.append(df)
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        # Split based on the 70/20/10 ratio established in the study [cite: 246]
        df_train_val = pd.concat([
            process_files(train_files, is_valid=False),
            process_files(val_files, is_valid=True)
        ], ignore_index=True)
        
        df_test = process_files(test_files, is_test=True) if test_files else pd.DataFrame()
        
        return df_train_val, df_test

    def sanitize_masks(self, df, class_pixel_value=40, sanitized_value=1):
        """
        Convert multi-class masks into binary masks for a target class.
        Original values: 40 (Crack), 160 (Water), 200 (Leaching)[cite: 194].
        """
        image_abs_paths, sanitized_paths, valid_indices = [], [], []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Sanitizing Masks"):
            try:
                # Construct absolute image path
                clean_filename = row['filename'].split('../')[-1]
                abs_img_path = os.path.normpath(os.path.join(self.base_dir, clean_filename))
                img_name = os.path.splitext(os.path.basename(abs_img_path))[0]
                
                # Derive mask name: TA_001_A -> TA_001_fuse_A_1band.png
                parts = img_name.rsplit('_', 1)
                mask_fn = f"{parts[0]}_fuse_{parts[1]}_1band.png" if len(parts) == 2 else f"{img_name}.png"
                
                raw_path = os.path.join(self.original_mask_dir, mask_fn)
                clean_path = os.path.join(self.sanitized_mask_dir, mask_fn)

                # Process raw mask if sanitized version does not exist
                if not os.path.exists(clean_path):
                    if not os.path.exists(raw_path):
                        print(f"Can't find raw path, skip {raw_path}")
                        continue
                    
                    mask_arr = np.array(Image.open(raw_path))
                    new_mask = np.zeros_like(mask_arr, dtype=np.uint8)
                    
                    # Target only the requested class pixels
                    if int(row.get('target', 0)) == 1:
                        new_mask[mask_arr == class_pixel_value] = sanitized_value
                    
                    Image.fromarray(new_mask).save(clean_path)

                image_abs_paths.append(abs_img_path)
                sanitized_paths.append(clean_path)
                valid_indices.append(idx)
                
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

        # Return updated dataframe with absolute paths
        df_clean = df.iloc[valid_indices].copy()
        df_clean['image_abs_path'] = image_abs_paths
        df_clean['mask_path_sanitized'] = sanitized_paths
        return df_clean

    def calculate_training_stats(self, df):
        """
        Calculate mean and standard deviation across the training set for custom normalization.
        This avoids using ImageNet defaults for specialized tunnel imagery.
        """
        print("Calculating custom dataset statistics...")
        # Only use training rows (is_valid == False and is_test == False)
        train_df = df[(df['is_valid'] == False) & (df['is_test'] == False)]
        paths = train_df['image_abs_path'].values
        
        means, stds = [], []
        for p in tqdm(paths, desc="Computing Stats"):
            # Normalize pixel values to [0, 1] before calculation
            img = np.array(Image.open(p).convert('RGB')) / 255.0
            means.append(np.mean(img, axis=(0, 1)))
            stds.append(np.std(img, axis=(0, 1)))
            
        return torch.tensor(np.mean(means, axis=0)), torch.tensor(np.mean(stds, axis=0))

    def get_dataloaders(self, train_val_df, test_df=None, bs=16, img_size=512, custom_stats=None):
        """
        Build fastai DataLoaders for train, validation, and (optionally) test sets.
        'img_size' resizes the high-res 2448x2048 images to uniform dimensions for batching[cite: 77, 191].
        """
        codes = np.array(['background', 'defect'])
        # Use custom stats if provided, otherwise default to ImageNet
        norm_stats = custom_stats if custom_stats else imagenet_stats
        print(norm_stats)

        dblock = DataBlock(
            blocks=(ImageBlock, MaskBlock(codes)),
            get_x=ColReader('image_abs_path'),
            get_y=ColReader('mask_path_sanitized'),
            splitter=ColSplitter('is_valid'),
            item_tfms=Resize(img_size), # Resizes high-res originals to uniform squares
            batch_tfms=[
                *aug_transforms(flip_vert=True, max_rotate=15.0, max_zoom=1.1, max_lighting=0.2),
                Normalize.from_stats(*norm_stats)
            ]
        )
        
        # Build training and validation loaders
        dls = dblock.dataloaders(train_val_df, bs=bs, num_workers=0, pin_memory=True)
        
        # Build test loader if test data is provided
        test_dl = None
        if test_df is not None and not test_df.empty:
            test_dl = dls.test_dl(test_df, with_labels=True)
            
        return dls.train, dls.valid, test_dl

if __name__ == "__main__":
    # 1. Define Paths relative to your project root (CSCI5527-FINAL)
    # We place 'sanitized_masks' at the root level, outside the dataset folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_folder = os.path.join(base_dir, "TACK_Tunnel_Data")
    
    csv_source_dir = os.path.join(dataset_folder, "2_model_input")
    raw_mask_dir = os.path.join(dataset_folder, "3_mask")

    # 2. Initialize the Pipeline
    # The class will automatically create 'sanitized_masks' in your project root
    pipeline = TunnelDataPipeline(
        base_dir=dataset_folder,
        original_mask_dir=raw_mask_dir
    )

    # 3. Load Dataset CSVs
    # Combining all three tunnels (TA, TB, TC) for a multi-domain study
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

    # 4. Sanitize Masks
    # Extracts Cracks (pixel value 40) and saves them as 0/1 binary masks 
    # in the 'sanitized_masks' directory outside the dataset folder
    print("Sanitizing training and validation masks...")
    df_train_val_ready = pipeline.sanitize_masks(df_train_val, class_pixel_value=40)
    
    print("Sanitizing test masks...")
    df_test_ready = pipeline.sanitize_masks(df_test, class_pixel_value=40)

    # 5. Calculate Custom Normalization Statistics
    # Uses actual tunnel imagery instead of default ImageNet stats
    custom_stats = pipeline.calculate_training_stats(df_train_val_ready)

    # 6. Generate Dataloaders
    # Returns the finalized fastai loaders for the training loop
    print("Generating Dataloaders...")
    train_dl, val_dl, test_dl = pipeline.get_dataloaders(
        train_val_df=df_train_val_ready,
        test_df=df_test_ready,
        bs=16,
        img_size=512,
        custom_stats=custom_stats
    )

    # Summary of the loaded data
    print(f"\nPipeline Ready:")
    print(f" - Training batches: {len(train_dl)}")
    print(f" - Validation batches: {len(val_dl)}")
    print(f" - Testing batches: {len(test_dl)}")

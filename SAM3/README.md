# SAM3 TTD Crack Segmentation Workspace

## TL;DR: Reproduce the SAM3 Results

Before running the commands, go to the SAM3 Hugging Face repo and request checkpoint access:

- https://huggingface.co/facebook/sam3

After access is approved, authenticate with Hugging Face and clone the official SAM3 repo next to this `CSCI5527-final` repo:

```bash
# 1. Start from the cloned course project repo.
cd CSCI5527-final

# 2. Clone the official SAM3 repo as a sibling of CSCI5527-final.
cd ..
git clone https://github.com/facebookresearch/sam3.git sam3

# 3. Log in to Hugging Face so SAM3 can download the gated checkpoints.
#    Create an access token at https://huggingface.co/settings/tokens first.
python -m pip install -U "huggingface_hub[cli]"
hf auth login

# 4. Install SAM3. Use the CUDA/PyTorch command that matches your machine if needed.
cd sam3
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[train,dev,notebooks]"

# 5. Apply this project's SAM3 source-code patch to the cloned SAM3 repo.
git apply ../CSCI5527-final/SAM3/reproducibility/sam3_patches/sam3_ttd_changes.patch

# 6. Copy the project-specific SAM3 YAML configs into the cloned SAM3 repo.
mkdir -p sam3/train/configs/ttd_fewshot
cp ../CSCI5527-final/SAM3/reproducibility/sam3_configs/ttd_fewshot/*.yaml \
   sam3/train/configs/ttd_fewshot/
```

Then run the notebooks in this order:

```text
CSCI5527-final/SAM3/sam3_ttd_zero_shot.ipynb
CSCI5527-final/SAM3/sam3_ttd_fewshot_setup.ipynb
CSCI5527-final/SAM3/sam3_ttd_fewshot_metric_conversion.ipynb
```

The final report-ready comparison table is:

```text
CSCI5527-final/SAM3/results/sam3_vs_unet_pixel_metric_comparison.csv
```

Notes:

- The `sam3/` directory is the separately cloned SAM3 repo after applying our patch. It is not committed inside this course project repo.
- The notebooks use relative path detection. If the external SAM3 repo is not at `<workspace>/sam3`, set `SAM3_REPO_ROOT`.
- Large generated outputs, checkpoints, prepared few-shot data, and per-image diagnostic CSVs are intentionally not tracked by git.
- SAM3 outputs COCO-style instance predictions, so the metric conversion notebook converts them to dense binary masks before comparing against U-Net pixel-level IoU/F1.

This directory contains the SAM3 experiments for crack segmentation on the TACK Tunnel Data (TTD) dataset. The experiments cover two settings:

- **SAM3 zero-shot**: run SAM3 directly with text prompts, without fine-tuning.
- **SAM3 few-shot fine-tuning**: fine-tune SAM3 with 5 / 10 / 25 / 50 shots per class, then evaluate on the test split and convert the results to pixel-level metrics for comparison with U-Net baselines.

## Expected Workspace Layout

For training/evaluation, this project expects the course project repo and the external SAM3 repo to be siblings:

```text
<workspace>/
  CSCI5527-final/
    SAM3/
  sam3/
```

The notebooks now infer paths from the current working directory instead of using hard-coded absolute paths. If your SAM3 repo is not located at `<workspace>/sam3`, set `SAM3_REPO_ROOT` before running the setup cell:

```python
import os
os.environ["SAM3_REPO_ROOT"] = "/path/to/sam3"
```

## Directory Overview

```text
CSCI5527-final/SAM3/
  README.md
  sam3_ttd_zero_shot.ipynb
  sam3_ttd_fewshot_setup.ipynb
  sam3_ttd_fewshot_metric_conversion.ipynb
  smoke_test.py
  ttd_dataset.py
  ttd_metrics.py
  generated_configs/
  fewshot_data/                 # ignored by git; generated/prepared data
  finetune_data/                # ignored by git
  outputs/                      # ignored by git; generated experiment outputs
  reproducibility/
  results/
```

## Main Files

`sam3_ttd_zero_shot.ipynb`

Runs the SAM3 zero-shot baseline using text prompts such as `crack`. This produces per-image and summary CSV files under `SAM3/outputs/zero_shot/`.

`sam3_ttd_fewshot_setup.ipynb`

Runs SAM3 few-shot fine-tuning and test evaluation. It supports both single-experiment execution and batch execution over all stable few-shot settings.

`sam3_ttd_fewshot_metric_conversion.ipynb`

Converts SAM3 COCO-style instance predictions into dense binary masks so that SAM3 can be compared with U-Net using the same pixel-level IoU, F1, precision, and recall definitions. It also merges SAM3 zero-shot, SAM3 few-shot, and U-Net baseline metrics into a final comparison table.

`ttd_dataset.py`

Utility code for reading the original TTD CSV splits, locating image/mask files, and creating binary crack masks.

`ttd_metrics.py`

Simple binary segmentation metrics: IoU, precision, recall, and F1.

`smoke_test.py`

Quick check that the dataset utilities can read the Single-TB split.

## Few-Shot Data

`fewshot_data/` contains prepared COCO-style few-shot datasets for SAM3. This folder is generated data and is ignored by git.

```text
fewshot_data/
  fewshot_experiment_summary.csv
  run_all_local_train.sh
  run_all_local_eval.sh
  run_all_local_train_and_eval.sh
  Single-TB/
  Shift-TA_TC-to-TB_10pct/
  Shift-TA_TB-to-TC_10pct/
```

Each experiment has four shot settings:

```text
5_shot_per_class/
10_shot_per_class/
25_shot_per_class/
50_shot_per_class/
```

Each shot folder follows this structure:

```text
5_shot_per_class/
  fewshot_metadata.json
  run_local_commands.sh
  train/
    images/
    masks/
    _annotations.coco.json
    train_manifest.csv
    train_manifest.jsonl
  val/
    images/
    masks/
    _annotations.coco.json
    val_manifest.csv
    val_manifest.jsonl
  test/
    images/
    masks/
    _annotations.coco.json
    test_manifest.csv
    test_manifest.jsonl
```

Important files:

- `images/`: input images.
- `masks/`: binary crack masks.
- `_annotations.coco.json`: COCO annotations used by SAM3 training/evaluation.
- `manifest.csv/jsonl`: human-readable image/mask mapping.
- `run_local_commands.sh`: generated train/eval commands for that shot setting.

## Generated Outputs

`outputs/` stores generated experiment outputs and is ignored by git.

```text
outputs/
  zero_shot/
  fewshot_runs/
  fewshot_metric_conversion/
```

A typical few-shot run output looks like:

```text
outputs/fewshot_runs/<experiment_name>/<shot>_shot_per_class_<run_tag>/
  checkpoints/
    checkpoint.pt
  dumps/
    ttd/
      val/coco_predictions_segm.json
      test/coco_predictions_segm.json
  logs/
    log.txt
    train_stats.json
    val_stats.json
    best_stats.json
  tensorboard/
  config.yaml
  config_resolved.yaml
```

The most important generated prediction file is:

```text
dumps/ttd/test/coco_predictions_segm.json
```

## SAM3 YAML Configs

SAM3 training configs are consumed from the external SAM3 repo:

```text
<workspace>/sam3/sam3/train/configs/ttd_fewshot/
```

For reproducibility, this project stores a copy of the project-specific configs under:

```text
SAM3/reproducibility/sam3_configs/ttd_fewshot/
```

Current run tags:

- `stable_lowlr_v1`: recommended setting. It uses a lower learning rate and fewer epochs to reduce NaN loss / bad checkpoint issues.
- `clean_fp32_v1`: earlier comparison setting, kept for reference.

Example Single-TB 5-shot configs:

```text
configs/ttd_fewshot/ttd_single_tb_5shot_textseg_stable_lowlr_v1.yaml
configs/ttd_fewshot/ttd_single_tb_5shot_textseg_stable_lowlr_v1_test_eval.yaml
```

The notebooks pass these paths relative to the SAM3 repo root:

```bash
python -m sam3.train.train -c configs/ttd_fewshot/...
```

## Running Few-Shot Experiments

Open:

```text
SAM3/sam3_ttd_fewshot_setup.ipynb
```

Run the first cell, **Experiment Selection**, before any train/eval cells.

### Single Experiment Mode

Set `EXPERIMENT_KEY` in the first code cell, for example:

```python
EXPERIMENT_KEY = single_tb_5shot_stable_lowlr_v1
```

or:

```python
EXPERIMENT_KEY = "single_tb_5shot_stable_lowlr_v1"
```

Then run the helper cell and call:

```python
run_train(EXPERIMENT_KEY)
run_eval(EXPERIMENT_KEY)
```

### Batch Mode

The notebook also defines `BATCH_EXPERIMENT_KEYS`, which runs all stable settings in this order:

1. `Single-TB`: 5 / 10 / 25 / 50-shot
2. `Shift-TA_TC-to-TB_10pct`: 5 / 10 / 25 / 50-shot
3. `Shift-TA_TB-to-TC_10pct`: 5 / 10 / 25 / 50-shot

The order intentionally runs all Single-TB YAMLs first, because that ordering avoided the earlier dtype/gradient issues observed during setup.

Run the **Batch Train + Eval** cell to process all stable experiments. Existing checkpoints/prediction JSON files are skipped by default.

## Common Experiment Keys

Single-TB:

```python
single_tb_5shot_stable_lowlr_v1
single_tb_10shot_stable_lowlr_v1
single_tb_25shot_stable_lowlr_v1
single_tb_50shot_stable_lowlr_v1
```

TA + TC train, TB test:

```python
shift_ta_tc_to_tb_10pct_5shot_stable_lowlr_v1
shift_ta_tc_to_tb_10pct_10shot_stable_lowlr_v1
shift_ta_tc_to_tb_10pct_25shot_stable_lowlr_v1
shift_ta_tc_to_tb_10pct_50shot_stable_lowlr_v1
```

TA + TB train, TC test:

```python
shift_ta_tb_to_tc_10pct_5shot_stable_lowlr_v1
shift_ta_tb_to_tc_10pct_10shot_stable_lowlr_v1
shift_ta_tb_to_tc_10pct_25shot_stable_lowlr_v1
shift_ta_tb_to_tc_10pct_50shot_stable_lowlr_v1
```

## Metric Conversion and U-Net Comparison

SAM3 outputs COCO-style instance predictions, while U-Net outputs one dense binary mask per image. To compare them fairly, `sam3_ttd_fewshot_metric_conversion.ipynb` converts SAM3 predictions into dense masks by unioning all predicted instance masks above a score threshold:

```text
SAM3 instance masks -> union mask -> pixel-level IoU/F1/precision/recall
```

The conversion notebook produces:

```text
SAM3/outputs/fewshot_metric_conversion/
  sam3_fewshot_pixel_metric_summary_threshold_sweep.csv
  sam3_fewshot_pixel_metric_per_image_threshold_sweep.csv
  unet_resnet34_imagenet_positive_negative_test_metrics.csv
  sam3_vs_unet_pixel_metric_comparison.csv
```

The final report-ready comparison table is copied to a tracked folder:

```text
SAM3/results/sam3_vs_unet_pixel_metric_comparison.csv
```

For a strict final comparison, use a fixed threshold or validation-selected threshold. Test-set threshold selection should be treated as exploratory / upper-bound analysis.

## Quick Dataset Check

From the project repo:

```bash
cd <workspace>/CSCI5527-final/SAM3
python3 smoke_test.py
```

This checks that `ttd_dataset.py` can read the Single-TB data and locate sample image/mask paths.

## Reproducibility Bundle

The external SAM3 repo was lightly modified for this project. Instead of committing the full SAM3 repo into the course project, this repo stores:

```text
SAM3/reproducibility/
  sam3_patches/
    sam3_ttd_changes.patch
  sam3_configs/
    ttd_fewshot/*.yaml
```

To reproduce the SAM3 setup from a fresh SAM3 checkout, place the two repos as siblings:

```text
<workspace>/
  CSCI5527-final/
  sam3/
```

Then apply the patch and copy the configs:

```bash
cd <workspace>/sam3
git apply ../CSCI5527-final/SAM3/reproducibility/sam3_patches/sam3_ttd_changes.patch
mkdir -p sam3/train/configs/ttd_fewshot
cp ../CSCI5527-final/SAM3/reproducibility/sam3_configs/ttd_fewshot/*.yaml \
   sam3/train/configs/ttd_fewshot/
```

After that, run:

```text
SAM3/sam3_ttd_fewshot_setup.ipynb
SAM3/sam3_ttd_fewshot_metric_conversion.ipynb
```

Large generated files such as checkpoints, prediction JSON files, TensorBoard logs, and per-image sweep CSVs remain under `SAM3/outputs/` and are intentionally ignored by git.

## Notes on Git Tracking

The repo tracks the notebooks, reproducibility patch/configs, README, and final report-ready result table. It does not track:

- model checkpoints (`*.pt`, `*.pth`)
- generated SAM3 outputs under `SAM3/outputs/`
- prepared data folders (`fewshot_data/`, `finetune_data/`, `TACK_Tunnel_Data/`)
- large per-image diagnostic CSVs

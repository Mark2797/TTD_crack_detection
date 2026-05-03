# SAM3 TTD Crack Segmentation Workspace

這個資料夾是用來跑 TACK Tunnel Data (TTD) crack segmentation 的 SAM3 實驗。現在主要有兩條線：

- zero-shot: 不訓練 SAM3，直接用 prompt 做預測。
- few-shot fine-tuning: 用 5 / 10 / 25 / 50 shot 的資料微調 SAM3，再做 test evaluation 和 IoU 分析。

## 路徑總覽

```text
CSCI5527-final/SAM3/
  README.md
  sam3_ttd_fewshot_setup.ipynb
  sam3_ttd_zero_shot.ipynb
  smoke_test.py
  ttd_dataset.py
  ttd_metrics.py
  generated_configs/
  finetune_data/
  fewshot_data/
  outputs/
```

## 重要檔案

`sam3_ttd_fewshot_setup.ipynb`

目前最重要的 notebook。用來選實驗、跑 SAM3 few-shot fine-tuning、跑 test eval、最後做 IoU analysis。現在已經改成只需要改第一個 code cell 的 `EXPERIMENT_KEY`。

`sam3_ttd_zero_shot.ipynb`

zero-shot baseline 用的 notebook。這條線不做 training，通常用來當 fine-tuning 前的 baseline。

`ttd_dataset.py`

把原始 TTD CSV split 讀進來，找到 image 和 mask，並產生 sanitized mask。這比較像資料準備工具。

`ttd_metrics.py`

簡單的 binary segmentation metrics：IoU、precision、recall、F1。

`smoke_test.py`

快速測試 `ttd_dataset.py` 能不能讀到 Single-TB 的資料。

`generated_configs/`

早期產生或暫存的 config。目前 few-shot SAM3 training 主要不是看這裡，而是看 `/users/7/yu001011/csci5527/sam3/sam3/train/configs/ttd_fewshot/`。

## fewshot_data 是什麼

`fewshot_data/` 是已經整理成 SAM3 可以吃的 few-shot COCO 格式資料。

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

每個實驗下面都有 4 種 shot 數：

```text
5_shot_per_class/
10_shot_per_class/
25_shot_per_class/
50_shot_per_class/
```

每個 shot 資料夾裡面長這樣：

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

重點：

- `images/`: 圖片。
- `masks/`: 對應的 crack mask。
- `_annotations.coco.json`: SAM3 training / eval 會讀的 COCO annotation。
- `manifest.csv/jsonl`: 比較方便人工檢查每張圖和 mask 的對應。
- `run_local_commands.sh`: 這個資料夾對應的 train/eval 指令。

`fewshot_experiment_summary.csv` 是總表，記錄每個 experiment + shot 對應到哪個 dataset root、output dir、train YAML、test eval YAML。

## outputs 是什麼

`outputs/` 放實驗結果。

```text
outputs/
  zero_shot/
  fewshot_runs/
```

few-shot training 的輸出會放在：

```text
outputs/fewshot_runs/<experiment_name>/<shot>_shot_per_class_<run_tag>/
```

例如：

```text
outputs/fewshot_runs/Shift-TA_TB-to-TC_10pct/5_shot_per_class_stable_lowlr_v1/
```

裡面通常會有：

```text
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

重點：

- `checkpoints/checkpoint.pt`: training 後的 checkpoint。
- `dumps/ttd/test/coco_predictions_segm.json`: test eval 的 segmentation prediction。
- `logs/`: 訓練和驗證 log。
- `tensorboard/`: TensorBoard event files。
- `config_resolved.yaml`: 實際跑 training 時 resolve 完的設定。

## YAML config 在哪

SAM3 的 training YAML 不在這個 `CSCI5527-final/SAM3` 資料夾裡，而是在 SAM3 repo 裡：

```text
/users/7/yu001011/csci5527/sam3/sam3/train/configs/ttd_fewshot/
```

每個實驗現在都有兩種 run tag：

- `stable_lowlr_v1`: 建議先用。學習率較低、epoch 較少，比較不容易出現 NaN loss 或壞 checkpoint。
- `clean_fp32_v1`: 舊版/比較用設定。保留給對照，不建議當預設。

例如 Single-TB 5-shot 建議用：

```text
configs/ttd_fewshot/ttd_single_tb_5shot_textseg_stable_lowlr_v1.yaml
configs/ttd_fewshot/ttd_single_tb_5shot_textseg_stable_lowlr_v1_test_eval.yaml
```

在 notebook 裡，這些 config 是用相對於 SAM3 repo 的路徑傳給 trainer：

```text
python -m sam3.train.train -c configs/ttd_fewshot/...
```

一般不用重新寫 YAML。現在 Single-TB、Shift-TA_TC-to-TB_10pct、Shift-TA_TB-to-TC_10pct 的 5 / 10 / 25 / 50 shot 都有 `stable_lowlr_v1` train/eval YAML。

## 怎麼跑 few-shot 實驗

打開：

```text
sam3_ttd_fewshot_setup.ipynb
```

第一個 code cell 是 `Experiment Selection`。只改這個變數：

```python
EXPERIMENT_KEY = single_tb_5shot_stable_lowlr_v1
```

也可以用字串寫法：

```python
EXPERIMENT_KEY = "single_tb_5shot_stable_lowlr_v1"
```

然後依序執行：

1. `Experiment Selection`
2. `Train`
3. `Test Eval`
4. `IoU Analysis`

`Experiment Selection` 會自動設定：

- `TRAIN_CONFIG`
- `TEST_CONFIG`
- `RUN_ROOT`
- `RUN_DIR`
- `CHECKPOINT`
- `GT_JSON`
- `PRED_JSON`

後面的 cell 都會共用這些變數。

## 常用 EXPERIMENT_KEY

Single-TB stable：

```python
EXPERIMENT_KEY = single_tb_5shot_stable_lowlr_v1
EXPERIMENT_KEY = single_tb_10shot_stable_lowlr_v1
EXPERIMENT_KEY = single_tb_25shot_stable_lowlr_v1
EXPERIMENT_KEY = single_tb_50shot_stable_lowlr_v1
```

TA + TC train, TB test stable：

```python
EXPERIMENT_KEY = shift_ta_tc_to_tb_10pct_5shot_stable_lowlr_v1
EXPERIMENT_KEY = shift_ta_tc_to_tb_10pct_10shot_stable_lowlr_v1
EXPERIMENT_KEY = shift_ta_tc_to_tb_10pct_25shot_stable_lowlr_v1
EXPERIMENT_KEY = shift_ta_tc_to_tb_10pct_50shot_stable_lowlr_v1
```

TA + TB train, TC test stable：

```python
EXPERIMENT_KEY = shift_ta_tb_to_tc_10pct_5shot_stable_lowlr_v1
EXPERIMENT_KEY = shift_ta_tb_to_tc_10pct_10shot_stable_lowlr_v1
EXPERIMENT_KEY = shift_ta_tb_to_tc_10pct_25shot_stable_lowlr_v1
EXPERIMENT_KEY = shift_ta_tb_to_tc_10pct_50shot_stable_lowlr_v1
```

舊版 clean 設定也還在，例如：

```python
EXPERIMENT_KEY = single_tb_5shot_clean_fp32_v1
```

但如果你只是要順利跑實驗，先用 `stable_lowlr_v1`。

## Single-TB 要怎麼跑

如果你現在要跑 Single-TB 5-shot，在 notebook 第一個 cell 用：

```python
EXPERIMENT_KEY = single_tb_5shot_stable_lowlr_v1
```

Notebook 會自動選到：

```text
configs/ttd_fewshot/ttd_single_tb_5shot_textseg_stable_lowlr_v1.yaml
configs/ttd_fewshot/ttd_single_tb_5shot_textseg_stable_lowlr_v1_test_eval.yaml
```

輸出會放到：

```text
outputs/fewshot_runs/Single-TB/5_shot_per_class_stable_lowlr_v1/
```

## 如果只想用 shell 跑

每個 few-shot dataset folder 裡都有 `run_local_commands.sh`，例如：

```text
fewshot_data/Single-TB/5_shot_per_class/run_local_commands.sh
```

裡面通常有兩行：

```bash
python -m sam3.train.train -c configs/ttd_fewshot/ttd_single_tb_5shot_textseg_stable_lowlr_v1.yaml --use-cluster 0 --num-gpus 1
python -m sam3.train.train -c configs/ttd_fewshot/ttd_single_tb_5shot_textseg_stable_lowlr_v1_test_eval.yaml --use-cluster 0 --num-gpus 1
```

有些 `run_local_commands.sh` 可能還是舊的 `clean_fp32_v1` 指令；建議以 notebook 顯示的 `TRAIN_CONFIG` / `TEST_CONFIG` 為準。

要注意這些指令要在 SAM3 repo root 跑：

```bash
cd /users/7/yu001011/csci5527/sam3
```

Notebook 已經幫你處理 `cwd`，所以比較不容易跑錯位置。

## 快速檢查資料讀取

可以跑：

```bash
cd /users/7/yu001011/csci5527/CSCI5527-final/SAM3
python3 smoke_test.py
```

它會測試 `ttd_dataset.py` 能不能讀到 Single-TB 的資料，並印出 sample image 和 mask path。

## 常見問題

### 我要跑 Single-TB，要重寫 YAML 嗎？

不用。直接改 notebook 裡的 `EXPERIMENT_KEY`，建議先選 `stable_lowlr_v1`。

### Train cell 說找不到 checkpoint 怎麼辦？

`Test Eval` 需要先有：

```text
RUN_DIR/checkpoints/checkpoint.pt
```

先跑 `Train` cell，或確認你選的 `EXPERIMENT_KEY` 對應的 `RUN_DIR` 裡已經有 checkpoint。

### IoU Analysis 找不到 prediction file 怎麼辦？

代表 test eval 還沒產生：

```text
dumps/ttd/test/coco_predictions_segm.json
```

先跑 `Test Eval` cell。

### 要改 learning rate 或 epoch 怎麼辦？

這種情況才需要改 YAML。YAML 在：

```text
/users/7/yu001011/csci5527/sam3/sam3/train/configs/ttd_fewshot/
```

建議複製一份新的 YAML，改新的 `run_tag` 和 output dir，不要直接覆蓋已經跑過的 config，這樣結果比較好追蹤。

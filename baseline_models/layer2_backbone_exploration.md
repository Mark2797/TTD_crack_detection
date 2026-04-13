# Layer 2 Backbone Exploration

This note defines the second exploration layer after the baseline analysis.

## Goal

The baseline analysis shows that the main weakness is not only imperfect mask boundaries, but also unstable feature representation on hard tunnel textures and domain-shift settings. In the worst-case examples, the model sometimes predicts crack-like structures at incorrect locations while missing the true crack region. This suggests that the next step should focus on the encoder/backbone rather than changing the full segmentation architecture immediately.

Therefore, Layer 2 keeps the segmentation architecture fixed as `Unet` and compares different backbones under the same pretrained setting.

## Fixed Setup

- Architecture: `Unet`
- Encoder weights: `imagenet`
- Image size: `512`
- Batch size: `16`
- Learning rate: `1e-4`
- Epochs: `100`
- Loss setting: keep the current baseline setting

## Target Experiment Settings

These three settings are the most informative for the second layer because they expose domain difficulty and domain-shift behavior.

1. `Single-TB`
2. `Shift-TA_TC-to-TB_10pct`
3. `Shift-TA_TB-to-TC_10pct`

## Candidate Backbones

### 1. `resnet34`

Role:
Baseline reference.

Why keep it:
This is the current baseline encoder, so it must remain in the comparison as the anchor model.

Expected use:
Provides the baseline point for every metric and every qualitative comparison.

### 2. `efficientnet-b0`

Role:
Main recommended Layer 2 candidate.

Why choose it:
The worst-case baseline visualizations suggest that the current model struggles with fine-grained texture discrimination, thin crack representation, and hard domain variation. EfficientNet is a reasonable next choice because it is often stronger at efficient "multi-scale feature extraction" than a standard ResNet encoder.

Expected improvement:
- Better `Test_IoU` and `Test_F1` on hard settings
- Better separation between crack and non-crack tunnel texture
- Fewer false positives in difficult background regions

### 3. `resnet50`

Role:
Deeper residual comparison model.

Why choose it:
This tests whether the improvement comes simply from a deeper, "higher-capacity encoder", or whether a different feature extraction strategy is actually needed.

Expected improvement:
- Possible gain in domain robustness
- Stronger representation on more complex scenes
- Useful control point against `efficientnet-b0`

## Main Hypothesis

If the baseline weakness mainly comes from insufficient feature representation, then stronger encoders should improve performance on the hardest settings even when the decoder and training setup remain unchanged.

In particular:

- If `efficientnet-b0` improves hard-case performance, that supports the idea that better texture and scale representation is needed.
- If `resnet50` also improves but less than `efficientnet-b0`, that suggests the gain is not only from model depth but from encoder design.
- If neither improves much, then the next exploration layer should move toward architecture changes such as `FPN`, `DeepLabV3Plus`, or `UnetPlusPlus`.

## Metrics To Watch

Primary metrics:

- `Test_IoU`
- `Test_F1`

Secondary metrics:

- `Test_Recall`
- `Test_Prec`

How to interpret them:

- Higher `Recall` with lower `Precision` means the model finds more crack pixels but may over-predict.
- Higher `Precision` with low `Recall` means the model is conservative and may miss thin cracks.
- The most convincing improvement is when `IoU` and `F1` improve together on the hard settings.



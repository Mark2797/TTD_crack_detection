# Explainability Suite

This folder contains a failure-driven explainability workflow for crack segmentation models. The main goal is not only to visualize model outputs, but to explain where the model succeeds or fails, what type of visual evidence causes errors, and whether those explanation signals can guide post-processing.

The suite is organized as a sequence of notebooks:

| Notebook | Purpose | Main output |
|---|---|---|
| `01_failure_case_taxonomy.ipynb` | Build a semi-automatic failure taxonomy from model predictions, ground truth masks, probability maps, image statistics, and connected-component geometry. | `failure_taxonomy_*.csv`, representative failure panels |
| `02_manual_review_viewer.ipynb` | Create review panels and audit sheets so the automatic failure labels can be manually checked. | `manual_review_*.csv`, `manual_review_panels/` |
| `03_uncertainty_error_maps.ipynb` | Analyze whether uncertainty and probability statistics explain different failure types. | `uncertainty_by_failure_type.csv`, uncertainty figures |
| `04_xai_guided_filtering.ipynb` | Test whether explanation-derived signals can filter unreliable predictions. | `xai_guided_filtering_*.csv`, filtering comparison figure |
| `05_connected_component_postprocessing_comparison.ipynb` | Evaluate validation-tuned connected-component post-processing and compare raw vs filtered predictions. | `connected_components_*_test_summary.csv`, delta tables, post-processing figures |

`00_method_roadmap.ipynb` is the high-level planning notebook. It explains the intended sequence of explainability methods and motivates why the project uses failure-driven analysis rather than only heatmaps.

## Methodological Framing

This suite treats explainability as a diagnostic pipeline:

1. Compute pixel-level error regions: true positives, false positives, and false negatives.
2. Measure model confidence and uncertainty in those regions.
3. Measure image and shape properties that may explain mistakes.
4. Assign candidate failure labels using transparent rules.
5. Generate representative examples and manually audit them.
6. Use explanation signals to test post-processing improvements.

This is a post-hoc, output-level explainability approach. It does not require changing the model architecture. Instead, it explains model behavior using the relationship between the input image, the ground truth mask, the prediction mask, the probability map, and connected-component morphology.

## 01 Failure Case Taxonomy

`01_failure_case_taxonomy.ipynb` is the core notebook. It runs a trained segmentation model, compares predictions against ground truth, and creates one row per image containing performance metrics and explanation features.

For each image, the notebook computes:

| Feature group | Columns / signals | Interpretation |
|---|---|---|
| Pixel confusion | `tp_pixels`, `fp_pixels`, `fn_pixels` | Where the model is correct, over-predicting, or missing cracks |
| Segmentation metrics | `iou`, `f1`, `precision`, `recall` | Overall quality of each prediction |
| Probability signals | `mean_prob_on_gt`, `mean_prob_on_fp` | Whether the model is confident on true cracks or false positives |
| Uncertainty | `mean_uncertainty` | Whether predictions are generally uncertain |
| Brightness | `image_brightness`, `fp_brightness`, `fn_brightness` | Whether errors occur in darker/brighter image regions |
| Edge strength | `image_edge_strength`, `fp_edge_strength`, `fn_edge_strength` | Whether false positives align with strong edges or seams |
| Non-crack overlap | `fp_overlap_water_pixels`, `fp_overlap_leaching_pixels`, `fp_overlap_noncrack_ratio` | Whether FP pixels overlap known non-crack defect labels |
| Component geometry | `pred_components`, `pred_largest_area`, `pred_max_elongation`, `gt_components`, `gt_largest_area`, `gt_max_elongation` | Whether cracks or predictions are fragmented, small, or long and line-like |

The notebook then assigns a `candidate_failure_type` using rule-based criteria. These labels are intentionally called "candidate" labels because they are generated automatically and should be manually reviewed before making strong claims.

## How Failure Types Are Detected

The failure taxonomy is based on interpretable rules. Each rule checks a measurable condition in the prediction, ground truth, probability map, or image statistics.

| Candidate label | Rule | Meaning |
|---|---|---|
| `missed_low_confidence_crack` | `fn_pixels > 100` and `mean_prob_on_gt < 0.35` | There is a real crack region, but the model assigns low crack probability to the ground truth crack pixels. This suggests the crack is missed because the model is under-confident. |
| `fragmented_or_thin_crack` | `fn_pixels > 100` and `gt_components >= 3` | The ground truth crack mask is split into several components and many crack pixels are missed. This suggests the model struggles with fragmented or thin crack structures. |
| `fp_overlaps_other_defect_mask` | `fp_pixels > 100` and `fp_overlap_noncrack_ratio >= 0.25` | At least 25% of false-positive pixels overlap non-crack defect masks such as water/leaching labels. This suggests the model confuses other tunnel defects with cracks. |
| `fp_dark_shadow_candidate` | `fp_pixels > 100` and `fp_brightness < image_brightness - 0.08` | False-positive regions are noticeably darker than the image average. This suggests shadow-like regions may be mistaken for cracks. |
| `fp_edge_or_seam_candidate` | `fp_pixels > 100` and `fp_edge_strength > image_edge_strength * 1.5` | False positives lie in areas with much stronger edges than the image average. This suggests the model may confuse seams, boundaries, or high-contrast edges with cracks. |
| `fp_long_linear_structure_candidate` | `fp_pixels > 100` and `pred_max_elongation > 20` | The predicted false-positive structure is very elongated. This suggests the model may detect long linear non-crack structures as cracks. |
| `over_thick_or_oversegmented_prediction` | `tp_pixels > 100` and `fp_pixels > tp_pixels * 0.5` | The model overlaps the real crack but also predicts too much surrounding area. This suggests the crack is detected but too thick or over-segmented. |
| `good_or_acceptable_prediction` | No failure rule fires and `iou >= 0.50` | The prediction is acceptable under the selected IoU threshold. |
| `mixed_or_unclear_failure` | No rule above applies | The error does not match a clear automatic pattern and should be manually inspected. |

Because one image can satisfy multiple rules, `candidate_failure_type` may contain multiple labels separated by commas. For example, a prediction can be both `fp_dark_shadow_candidate` and `fp_long_linear_structure_candidate`.

The thresholds are heuristic but transparent. Their role is to generate candidate explanations and representative examples, not to replace manual inspection.

<!-- ## Threshold Sweep

`01_failure_case_taxonomy.ipynb` also includes a threshold sweep over crack probability thresholds. This checks how IoU, F1, precision, recall, FP pixels, and FN pixels change as the decision threshold changes.

Important interpretation:

- The sweep is useful as sensitivity analysis.
- A threshold should not be selected using test-set performance for final reporting.
- For fair evaluation, choose the threshold on validation data, fix it, and then evaluate once on the test set.
- If a test-set sweep is shown, it should be described as an oracle diagnostic or sensitivity plot, not as final model selection. -->

## 02 Manual Review Viewer

`02_manual_review_viewer.ipynb` supports human auditing. The automatic taxonomy is useful, but some labels are necessarily approximate because image artifacts can be ambiguous.

This notebook creates review panels and CSV sheets for manual labeling. The intended workflow is:

1. Load representative samples from each candidate failure type.
2. Display image, ground truth, prediction, and error regions.
3. Fill `human_failure_type` and `notes`.
4. Use the audited labels to avoid overclaiming the automatic taxonomy.

<!-- This step is important because it turns the taxonomy into a semi-automatic analysis: the rules select candidates, and human review validates the interpretation. -->

## 03 Uncertainty and Error Maps

`03_uncertainty_error_maps.ipynb` asks whether uncertainty and probability statistics explain different types of failure.

It aggregates the taxonomy output by `model_name` and `failure_type`, then reports:

- number of samples per failure type
- mean IoU / F1 / precision / recall
- mean uncertainty
- mean probability on ground truth crack pixels
- mean probability on false-positive pixels
- total FP and FN pixels

It also bins images by uncertainty quantile and checks whether higher uncertainty corresponds to lower performance.

Useful interpretation:

- High `mean_uncertainty` with low IoU suggests the model is uncertain where it fails.
- Low `mean_prob_on_gt` for missed cracks supports the `missed_low_confidence_crack` explanation.
- High `mean_prob_on_fp` suggests confident false positives, which are more concerning than uncertain false positives.

## 04 XAI-Guided Filtering

`04_xai_guided_filtering.ipynb` tests whether explanation signals can improve prediction quality through filtering.

The current notebook performs image-level filtering using:

- `mean_uncertainty`
- `mean_prob_on_fp`
- `pred_components`

The idea is to reject or filter unreliable predictions when they show explanation patterns associated with failure. The notebook compares baseline metrics with filtered metrics and saves the sweep results.

This is an exploratory step. It is useful for showing that explanation signals can guide interventions, but component-level filtering is a stronger future extension because crack segmentation errors often happen at the component level rather than the whole-image level.

## 05 Connected-Component Post-Processing Comparison

`05_connected_component_postprocessing_comparison.ipynb` evaluates a more concrete post-processing rule: remove predicted crack components smaller than a selected area.

The method is:

1. Predict on the validation split and test split using a fixed threshold.
2. Sweep candidate `min_component_area` values on validation predictions.
3. Select the area threshold using the validation metric, usually `mean_f1`.
4. Apply that fixed area threshold to the test split.
5. Compare raw predictions against post-processed predictions.

This avoids test-set leakage because the component-area threshold is selected only on validation data. The test set is used only after the rule is fixed.

The key output is the comparison between:

- `postprocess = none`
- `postprocess = remove_small_components`

Important metrics:

| Metric | What it tells us |
|---|---|
| `mean_iou` | Average per-image IoU. Each image has equal weight. |
| `micro_iou` | Dataset-level IoU after summing TP/FP/FN pixels across all images. Large cracks/images have more influence. |
| `fp_pixels_delta` | How many false-positive pixels were removed. Negative is good. |
| `fn_pixels_delta` | How many false-negative pixels were added. Positive means the filter removed some true crack pixels. |
| `micro_precision` | Among all predicted crack pixels, how many are correct. Usually improves when FP is removed. |
| `micro_recall` | Among all true crack pixels, how many are detected. Can drop if real thin cracks are removed. |

Typical interpretation:

- If FP pixels decrease and precision increases, the filter is removing false-positive fragments.
- If FN pixels increase and recall drops, the filter is too aggressive and removes real cracks.
- If `mean_iou` improves but `micro_iou` does not, the filter may help many small/easy images while not improving the dataset-level pixel balance.
- If `micro_iou` improves, the post-processing rule improves the overall pixel-level segmentation result.

## Outputs

Most generated files are stored under `outputs/`.

Common output files:

| File pattern | Meaning |
|---|---|
| `failure_taxonomy_*.csv` | Per-image taxonomy and explanation features |
| `threshold_sweep_*.csv` | Threshold sensitivity results |
| `representative_failure_samples_*.csv` | Selected examples for each failure type |
| `representative_failure_panels_*/` | Visual panels for representative failures |
| `manual_review_*.csv` | Manual audit sheets |
| `uncertainty_by_failure_type.csv` | Aggregated uncertainty by failure category |
| `xai_guided_filtering_*.csv` | Filtering sweep results |
| `postprocessing/connected_components_*_test_summary.csv` | Raw vs connected-component post-processing comparison |
| `postprocessing/connected_components_*_metadata.json` | Selected validation area threshold and experiment metadata |

## Recommended Reporting Language

One concise way to describe the method:

> I use a failure-driven explainability pipeline for crack segmentation. First, I compute TP, FP, and FN regions for each prediction. Then I combine pixel-level errors with model probability, uncertainty, brightness, edge strength, non-crack defect overlap, and connected-component geometry. These signals are used to assign candidate failure types such as missed low-confidence cracks, fragmented thin cracks, false positives on shadows, false positives on strong edges, and over-segmented predictions. Representative examples are then manually reviewed to avoid overclaiming. Finally, explanation signals are used to test post-processing rules, such as validation-tuned connected-component filtering, to evaluate whether the explanations can lead to measurable improvements.

## Caveats

- The taxonomy labels are heuristic candidate labels, not ground-truth causal labels.
- Manual review is needed before making strong claims about visual causes.
- Test-set threshold sweeps should be treated as sensitivity analysis, not final model selection.
- Connected-component filtering can reduce false positives but may increase false negatives, especially for thin cracks.
- `mean_iou` and `micro_iou` can tell different stories; both should be reported when discussing post-processing.

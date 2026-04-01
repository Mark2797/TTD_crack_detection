import torch

def _get_stats(inp, targ, class_idx=1, smooth=1e-6):
    """
    Calculates the base confusion matrix components for a specific class.
    
    Args:
        inp: Raw model logits of shape (N, C, H, W).
        targ: Ground truth mask of shape (N, 1, H, W) or (N, H, W).
        class_idx: The index of the defect class (default is 1 for cracks).
        smooth: Small constant to prevent division by zero errors.
        
    Returns:
        tp, fp, fn, tn, smooth.
    """
    # Convert logits to class indices by picking the channel with the highest value
    # Resulting shape: (N, H, W)
    pred = inp.argmax(dim=1).as_subclass(torch.Tensor)
    
    # Remove channel dimension from target to match prediction shape (N, H, W)
    targ = targ.as_subclass(torch.Tensor).squeeze(1)
    
    # Calculate True Positives: Predicted crack AND is actually a crack
    tp = ((pred == class_idx) & (targ == class_idx)).sum().float()
    
    # Calculate False Positives: Predicted crack BUT is actually background
    fp = ((pred == class_idx) & (targ != class_idx)).sum().float()
    
    # Calculate False Negatives: Predicted background BUT is actually a crack
    fn = ((pred != class_idx) & (targ == class_idx)).sum().float()
    
    # Calculate True Negatives: Predicted background AND is actually background
    tn = ((pred != class_idx) & (targ != class_idx)).sum().float()
    
    return tp, fp, fn, tn, smooth

def iou_crack(inp, targ):
    """
    Computes Intersection over Union (Jaccard Index) for the crack class.
    Formula: $IoU = \frac{TP}{TP + FP + FN}$
    """
    tp, fp, fn, _, smooth = _get_stats(inp, targ)
    return (tp + smooth) / (tp + fp + fn + smooth)

def dice_score_crack(inp, targ):
    """
    Computes the Dice Coefficient (Sørensen–Dice Index).
    Formula: $Dice = \frac{2TP}{2TP + FP + FN}$
    """
    tp, fp, fn, _, smooth = _get_stats(inp, targ)
    return (2 * tp + smooth) / (2 * tp + fp + fn + smooth)

def recall_crack(inp, targ):
    """
    Computes Recall (Sensitivity). Measures the ability to find all actual cracks.
    Formula: $Recall = \frac{TP}{TP + FN}$
    """
    tp, _, fn, _, smooth = _get_stats(inp, targ)
    return (tp + smooth) / (tp + fn + smooth)

def precision_crack(inp, targ):
    """
    Computes Precision. Measures how many predicted cracks are actually correct.
    Formula: $Precision = \frac{TP}{TP + FP}$
    """
    tp, fp, _, _, smooth = _get_stats(inp, targ)
    return (tp + smooth) / (tp + fp + smooth)

def f1_score_crack(inp, targ):
    """
    Computes the F1 Score, the harmonic mean of Precision and Recall.
    Useful for imbalanced data where background pixels outnumber crack pixels.
    Formula: $F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$
    """
    tp, fp, fn, _, smooth = _get_stats(inp, targ)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return 2 * (precision * recall) / (precision + recall + smooth)

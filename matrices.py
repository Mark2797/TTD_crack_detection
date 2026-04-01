def _get_stats(inp, targ, class_idx=1, smooth=1e-6):
    pred = inp.argmax(dim=1)
    targ = targ.squeeze(1)
    tp = ((pred == class_idx) & (targ == class_idx)).sum().float()
    fp = ((pred == class_idx) & (targ != class_idx)).sum().float()
    fn = ((pred != class_idx) & (targ == class_idx)).sum().float()
    tn = ((pred != class_idx) & (targ != class_idx)).sum().float()
    return tp, fp, fn, tn, smooth

def iou_crack(inp, targ):
    tp, fp, fn, _, smooth = _get_stats(inp, targ)
    return (tp + smooth) / (tp + fp + fn + smooth)

def dice_score_crack(inp, targ):
    tp, fp, fn, _, smooth = _get_stats(inp, targ)
    return (2 * tp + smooth) / (2 * tp + fp + fn + smooth)

def recall_crack(inp, targ):
    tp, _, fn, _, smooth = _get_stats(inp, targ)
    return (tp + smooth) / (tp + fn + smooth)

def precision_crack(inp, targ):
    tp, fp, _, _, smooth = _get_stats(inp, targ)
    return (tp + smooth) / (tp + fp + smooth)

def f1_score_crack(inp, targ):
    tp, fp, fn, _, smooth = _get_stats(inp, targ)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return 2 * (precision * recall) / (precision + recall + smooth)

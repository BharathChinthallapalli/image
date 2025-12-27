import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def compute_dice(pred_prob, gt_mask, threshold=0.5):
    """
    Computes Dice Coefficient.
    pred_prob: Continuous probability map [0,1]
    gt_mask: Binary Ground Truth
    """
    pred_mask = (pred_prob > threshold)
    intersection = (pred_mask & gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / union

def compute_iou(pred_prob, gt_mask, threshold=0.5):
    """
    Computes Jaccard Index (IoU).
    """
    pred_mask = (pred_prob > threshold)
    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    
    if union == 0:
        return 1.0
    return intersection / union

def compute_psnr(pred_prob, gt_mask):
    """
    Computes PSNR between probability map and binary GT.
    """
    # GT is binary [0, 1], Pred is [0, 1]
    # Treat GT as float image
    return psnr(gt_mask.astype(np.float32), pred_prob, data_range=1.0)

def compute_ssim(pred_prob, gt_mask):
    """
    Computes SSIM.
    """
    return ssim(gt_mask.astype(np.float32), pred_prob, data_range=1.0)

def compute_mse(pred_prob, gt_mask):
    """
    Computes Mean Squared Error.
    """
    return np.mean((pred_prob - gt_mask.astype(np.float32)) ** 2)

def evaluate_all(pred_prob, gt_mask):
    """
    Returns a dictionary of all metrics.
    """
    return {
        "Dice": compute_dice(pred_prob, gt_mask),
        "IoU": compute_iou(pred_prob, gt_mask),
        "PSNR": compute_psnr(pred_prob, gt_mask),
        "SSIM": compute_ssim(pred_prob, gt_mask),
        "MSE": compute_mse(pred_prob, gt_mask)
    }

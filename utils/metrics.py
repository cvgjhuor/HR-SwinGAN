import numpy as np
import torch
from skimage.metrics import structural_similarity


# Determine dynamic range for metric computation (defaults to max intensity of target)
def _safe_data_range(image: np.ndarray, maxval: float = None) -> float:
    """Determine valid dynamic range for metric computation."""
    data_range = float(maxval) if maxval is not None else float(np.max(image))
    if not np.isfinite(data_range) or data_range <= 0.0:
        return 1.0
    return data_range


# Normalized Mean Squared Error: ||gt - pred||_2^2 / ||gt||_2^2
def nmse(gt, pred) -> float:
    """Calculate Normalized Mean Squared Error between target and prediction."""
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    return float(np.sum((gt - pred) ** 2) / (np.sum(gt ** 2) + 1e-12))


# Peak Signal-to-Noise Ratio (dB): 10 * log10(MAX^2 / MSE)
def psnr(gt, pred, maxval: float = None) -> float:
    """Calculate Peak Signal-to-Noise Ratio (dB) between target and prediction."""
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    mse = float(np.mean((gt - pred) ** 2))
    if mse == 0.0:
        return float('inf')

    dyn_range = _safe_data_range(gt, maxval)
    return float(10.0 * np.log10((dyn_range ** 2) / mse))


# Structural Similarity Index (SSIM) averaged across batch slices
def ssim(gt, pred, maxval: float = None) -> float:
    """Calculate Structural Similarity Index across batch-level slices."""
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    if gt.ndim >= 3:
        batch_size = gt.shape[0]
        total_ssim = 0.0
        for i in range(batch_size):
            slice_gt = gt[i].squeeze()
            slice_pred = pred[i].squeeze()
            dyn_range = _safe_data_range(slice_gt, maxval)
            total_ssim += structural_similarity(slice_gt, slice_pred, data_range=dyn_range)
        return total_ssim / batch_size
    else:
        dyn_range = _safe_data_range(gt, maxval)
        return float(structural_similarity(gt.squeeze(), pred.squeeze(), data_range=dyn_range))


# Running average tracker for metrics and losses
class AverageMeter(object):
    def __init__(self):
        """Initialize running average accumulator."""
        self.reset()

    def reset(self):
        """Reset all accumulator statistics to zero."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """Update accumulator with new value and sample count."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

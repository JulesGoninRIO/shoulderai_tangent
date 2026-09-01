import torch
import torch.nn.functional as F
from torchmetrics import Metric
import numpy as np
from utils.utils_preproc import fit_line
from src.dataset import set_seed

class TangentSignLoss(Metric):
    def __init__(self, angle_loss=1, dist_penalty=1.0, mse_penalty=2.0):
        super().__init__()
        set_seed(42)
        self.angle_loss = angle_loss
        self.dist_penalty = dist_penalty
        self.continuity_penalty = 1.0
        self.mse_penalty = mse_penalty
        self.lenght_penalty = 0.3
        self.reset()

    def reset(self):
        self.dist = 0
        self.continuity_error = 0
        self.linearity_error = 0
        self.mse_error = 0
        self.lenght_error = 0

    def update(self, preds, targets, show=True):
        for pred, target in zip(preds, targets):
            if (pred == 0).all():
                if show:
                    print("no prediction")
                self.continuity_error += 1000000
                continue
            pred_line = fit_line(pred[0], loss = True, prediction=True)
            target_line = fit_line(target[0], loss = True)
            mse_loss = F.mse_loss(pred_line, target_line)
            self.mse_error += mse_loss



    def compute(self):
        self.total = self.continuity_error + self.mse_error

        # Make sure the total loss is a tensor
        if isinstance(self.total, (int, float)):
            self.total = torch.tensor(self.total, dtype=torch.float32)
        elif isinstance(self.total, torch.Tensor) and self.total.numel() > 1:
            self.total = self.total.mean()

        return self.total
    

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    # Flatten the arrays to 1D for comparison
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Compute intersection and union
    intersection = np.sum(y_true_flat * y_pred_flat)
    union = np.sum(y_true_flat) + np.sum(y_pred_flat)
    
    # Compute Dice coefficient
    dice = (2 * intersection + smooth) / (union + smooth)
    
    return dice
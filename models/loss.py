import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedMSEWithLogitsLoss(nn.Module):
    """
    MSE Loss that masks out values where target < 0
    
    Args:
        mask_before (bool): If True, applies mask before computing MSE.
                          If False, applies mask after computing MSE.
        reduction (str): Specifies the reduction to apply: 'none' | 'mean' | 'sum'
    """
    def __init__(self, mask_before=True, reduction='mean'):
        super().__init__()
        self.mask_before = mask_before
        self.reduction = reduction
        
    def forward(self, pred, target):
        # Create mask where target >= 0
        mask = (target >= 0).float()
        
        if self.mask_before:
            # Apply mask before computing MSE
            masked_pred = nn.Sigmoid(pred) * mask
            masked_target = target * mask
            loss = F.mse_loss(masked_pred, masked_target, reduction='none')
            
            if self.reduction == 'none':
                return loss
            elif self.reduction == 'mean':
                # Normalize by number of valid elements
                valid_elements = mask.sum()
                return loss.sum() / (valid_elements + 1e-8)
            else:  # sum
                return loss.sum()
        else:
            # Compute MSE first, then apply mask
            loss = F.mse_loss(pred, target, reduction='none')
            masked_loss = loss * mask
            
            if self.reduction == 'none':
                return masked_loss
            elif self.reduction == 'mean':
                valid_elements = mask.sum()
                return masked_loss.sum() / (valid_elements + 1e-8)
            else:  # sum
                return masked_loss.sum()
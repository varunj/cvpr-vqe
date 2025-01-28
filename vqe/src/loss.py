import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    @staticmethod
    def forward(y_hat: torch.Tensor, y_gt: torch.Tensor):
        return F.mse_loss(y_hat, y_gt, reduction='mean')

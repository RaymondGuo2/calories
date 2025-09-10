import torch.nn as nn
import torch

class RMSLELoss(nn.Module):
    def __init__(self):
        super(RMSLELoss, self).__init__()
    
    def forward(self, pred, target):
        pred = torch.clamp(pred, min=0)
        target = torch.clamp(target, min=0)
        return torch.sqrt(torch.mean((torch.log1p(pred) - torch.log1p(target)) ** 2))

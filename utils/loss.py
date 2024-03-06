import torch
from torch import nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Класс, реализующий Dice loss для бинарной 
    семантической сегментации. Используется при
    дисбалансе положительного класса.
    """
    def __init__(self, weight=None, smooth=1):
        super().__init__()
        self.smooth=smooth


    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs).view(-1)
        targets = targets.view(-1).float()
        intersection = torch.sum(inputs * targets)
        dice = (2. * intersection + self.smooth) / (torch.sum(inputs) + torch.sum(targets) + self.smooth) 
        return 1 - dice
    

class FocalLoss(nn.Module):
    """
    Класс, реализующий Focal loss для бинарной семантической
    сегментации. Используется при дисбалансе положительного класса.
    """
    def __init__(self, gamma=2, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.eps = 10e-6


    def forward(self, input, target):
        logpt = F.sigmoid(input)
        pt = (target == 0) * (1 - logpt) + target * logpt
        pt = torch.clamp(pt, min=self.eps, max=1.0)
        loss = - (1 - pt) ** self.gamma * torch.log(pt)
        return torch.mean(loss)
    
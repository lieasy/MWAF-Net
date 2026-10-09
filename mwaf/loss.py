"""Main-network training objective and Mixup."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Archived label-smoothed focal-style objective; pt uses smoothed cross-entropy."""

    def __init__(
        self, num_classes, gamma=2.5, alpha=None, reduction="mean", label_smoothing=0.2
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def forward(self, pred, target):
        if self.label_smoothing > 0:
            one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
            one_hot = (
                one_hot * (1 - self.label_smoothing)
                + self.label_smoothing / self.num_classes
            )
            log_prob = F.log_softmax(pred, dim=1)
            ce_loss = -(one_hot * log_prob).sum(dim=1)
        else:
            ce_loss = nn.CrossEntropyLoss(reduction="none")(pred, target)
        pt = torch.exp(-ce_loss)
        focal_loss = ce_loss * (1 - pt) ** self.gamma
        if self.alpha is not None:
            if isinstance(self.alpha, (list, np.ndarray)):
                alpha = torch.tensor(
                    self.alpha, dtype=torch.float32, device=pred.device
                )
            else:
                alpha = self.alpha
            alpha = alpha[target]
            focal_loss = alpha * focal_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def mixup_data(x, y, alpha=0.2):
    """Mixup增强：生成混合样本（适配多分类）"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = (y, y[index])
    return (mixed_x, y_a, y_b, lam)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """计算Mixup损失（适配多分类）"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

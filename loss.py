"""The smoothed focal-style objective in manuscript Section II-D."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def inverse_frequency_alpha(labels, num_classes):
    """Inverse frequencies with training-sample mean weight equal to one.

    This normalisation is an explicit implementation convention. Frequencies
    always come from the training partition before replacement sampling.
    """
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes)
    if len(counts) != num_classes or np.any(counts == 0):
        raise ValueError("Every class must occur in training to compute alpha")
    return len(labels) / (num_classes * counts.astype(np.float64))


class FocalLoss(nn.Module):
    """alpha[y] * (1 - exp(-smoothed_CE))**gamma * smoothed_CE.

    With nonzero smoothing, p_t is the smoothed-label geometric mean rather
    than the probability of the hard target. This matches the paper's
    focal-style equation rather than a different standard focal objective.
    """

    def __init__(self, num_classes, gamma=2.5, alpha=None, reduction="mean",
                 label_smoothing=0.2):
        super().__init__()
        if num_classes < 2 or gamma < 0 or not 0 <= label_smoothing < 1:
            raise ValueError("Invalid class count, gamma, or label smoothing")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be mean, sum, or none")
        self.num_classes = num_classes
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction
        alpha = torch.ones(num_classes) if alpha is None else torch.as_tensor(
            alpha, dtype=torch.float32
        )
        if alpha.shape != (num_classes,) or not torch.isfinite(alpha).all() or (alpha < 0).any():
            raise ValueError("alpha must contain one finite nonnegative weight per class")
        self.register_buffer("alpha", alpha.clone())

    def forward(self, inputs, targets):
        if inputs.ndim != 2 or inputs.shape[1] != self.num_classes:
            raise ValueError("Logits must have shape [batch, num_classes]")
        ce = F.cross_entropy(inputs, targets, reduction="none",
                             label_smoothing=self.label_smoothing)
        loss = self.alpha[targets] * (1 - torch.exp(-ce)).pow(self.gamma) * ce
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean() if self.reduction == "mean" else loss


class SampleMeanCrossEntropy(FocalLoss):
    """Smoothed CE with optional target-class alpha, averaged over samples.

    Gamma zero removes the focal factor exactly. Target-class weights multiply
    each complete smoothed CE term; the denominator is sample count, not the
    sum of weights in a mini-batch. Sample-count accumulation is then exact.
    """

    def __init__(self, num_classes, alpha=None, label_smoothing=0.0):
        super().__init__(num_classes, gamma=0.0, alpha=alpha,
                         label_smoothing=label_smoothing)


def mixup_data(x, y, alpha=0.4):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(len(x), device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

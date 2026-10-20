from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F

from .wavelets import generate_wavelet_kernel


MODEL_CONFIG_VERSION = "mwaf-manuscript-v2"


class DropBlock1D(nn.Module):
  
    def __init__(self, drop_prob=0.15, block_size=3, warmup_epochs=0):
        super().__init__()
        if not 0.0 <= drop_prob <= 1.0:
            raise ValueError("drop_prob must be between 0 and 1")
        if block_size < 1 or block_size % 2 != 1:
            raise ValueError("block_size must be a positive odd integer")
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be nonnegative")
        self.max_drop_prob = float(drop_prob)
        self.block_size = int(block_size)
        self.warmup_epochs = int(warmup_epochs)
        self.register_buffer("scheduled_prob", torch.tensor(0.0))
        self.set_epoch(0)

    @property
    def drop_prob(self):
        return float(self.scheduled_prob.item())

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        ramp = min(float(epoch) / self.warmup_epochs, 1.0) if self.warmup_epochs else 1.0
        self.scheduled_prob.fill_(self.max_drop_prob * ramp)

    def forward(self, x):
        if not self.training:
            return x
        gamma = self.scheduled_prob / self.block_size
        centres = (torch.rand(x.shape[0], 1, x.shape[-1], device=x.device) < gamma).to(x.dtype)
        dropped = F.max_pool1d(
            centres, kernel_size=self.block_size, stride=1,
            padding=self.block_size // 2,
        )
        keep = 1.0 - dropped
        kept = keep.sum(dtype=torch.float32)
        rescale = keep.numel() / (kept + 1e-8)
        rescale = torch.where(kept > 0, rescale, torch.zeros_like(rescale))
        return x * keep * rescale


class SqueezeExcitation1D(nn.Module):

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class MinorityClassFeatureEnhancement1D(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, 7, padding=3, groups=channels, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.gate = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return x + self.gate * F.relu(self.bn(self.depthwise(x)))


class MultiScaleWaveletConv1d(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel_size, wavelet="morl", stride=1,
        padding=0, dilation=1, bias=False, se_reduction=4,
    ):
        super().__init__()
        if out_channels < 3:
            raise ValueError("out_channels must be at least 3 for three scale groups")
        if se_reduction < 1:
            raise ValueError("se_reduction must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.wavelet = wavelet
        group_channels = (out_channels // 3, out_channels // 3,
                          out_channels - 2 * (out_channels // 3))
        for i, (channels, scale) in enumerate(zip(group_channels, (1.0, 2.0, 4.0)), 1):
            self.register_buffer(f"scale{i}", torch.full((channels, in_channels, 1), scale))
            self.register_buffer(f"shift{i}", torch.zeros(channels, in_channels, 1))
            setattr(self, f"conv{i}", nn.Conv1d(
                in_channels, channels, kernel_size, stride, padding, dilation,
                groups=1, bias=bias,
            ))
        self.se = SqueezeExcitation1D(out_channels, se_reduction)
        self.mcfe = MinorityClassFeatureEnhancement1D(out_channels)
        self._init_wavelet_kernels()

    def _init_wavelet_kernels(self):

        with torch.no_grad():
            for i in (1, 2, 3):
                conv = getattr(self, f"conv{i}")
                kernel = generate_wavelet_kernel(
                    self.wavelet, self.kernel_size,
                    getattr(self, f"scale{i}"), getattr(self, f"shift{i}"),
                    device=conv.weight.device,
                )
                conv.weight.copy_(kernel)
                if conv.bias is not None:
                    conv.bias.zero_()

    def forward(self, x):
        combined = torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], dim=1)
        return self.mcfe(self.se(combined))


class CoordAttention1D(nn.Module):

    def __init__(self, channels, reduction=4):
        super().__init__()
        if reduction < 1:
            raise ValueError("reduction must be positive")
        self.channels = channels
        self.reduction = reduction
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, hidden, 1)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.relu = nn.ReLU(inplace=True)
        self.fc_x = nn.Conv1d(hidden, channels, 1)
        self.fc_c = nn.Conv1d(hidden, channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.g_att = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        descriptor = self.relu(self.bn1(self.fc1(self.avg_pool(x))))
        att_x = self.sigmoid(self.fc_x(descriptor))
        att_c = self.sigmoid(self.fc_c(descriptor))
        mixing = torch.sigmoid(self.g_att)
        return x * (mixing * att_x + (1.0 - mixing) * att_c)


class EnhancedWaveletTCNBlock(nn.Module):

    def __init__(
        self, in_channels, out_channels, wavelet="morl", kernel_size=3,
        dilation=1, dropout=0.2, dropblock_prob=0.15,
        dropblock_warmup_epochs=15, dropblock_size=3,
        se_reduction=4, attention_reduction=4,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = MultiScaleWaveletConv1d(
            in_channels, out_channels, kernel_size, wavelet=wavelet,
            padding=padding, dilation=dilation, bias=False, se_reduction=se_reduction,
        )
        self.bn1 = nn.BatchNorm1d(out_channels, momentum=0.95)
        self.relu = nn.ReLU(inplace=True)
        self.dropblock = DropBlock1D(dropblock_prob, dropblock_size, dropblock_warmup_epochs)
        self.intermediate = nn.Conv1d(out_channels, out_channels, 1, bias=False)
        self.conv2 = MultiScaleWaveletConv1d(
            out_channels, out_channels, kernel_size, wavelet=wavelet,
            padding=padding, dilation=dilation, bias=False, se_reduction=se_reduction,
        )
        self.bn2 = nn.BatchNorm1d(out_channels, momentum=0.95)
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels, momentum=0.95),
            )
        self.ca = CoordAttention1D(out_channels, attention_reduction)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.dropblock(self.relu(self.bn1(self.conv1(x))))
        out = self.intermediate(out)
        out = self.dropout(self.ca(self.bn2(self.conv2(out))))
        return self.relu(out + residual)


class WaveletBranch(nn.Module):

    def __init__(
        self, wavelet, in_channels=1, branch_channels=(64, 64, 128),
        dropout=0.2, block_dropout=0.2, **block_kwargs,
    ):
        super().__init__()
        c0, c1, c2 = branch_channels
        self.wavelet = wavelet
        self.input_conv = nn.Sequential(
            MultiScaleWaveletConv1d(
                in_channels, c0, kernel_size=7, wavelet=wavelet,
                stride=2, padding=3, se_reduction=block_kwargs.get("se_reduction", 4),
            ),
            nn.BatchNorm1d(c0, momentum=0.95),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Dropout(dropout),
        )
        self.tcn1 = EnhancedWaveletTCNBlock(
            c0, c1, wavelet=wavelet, dilation=1, dropout=block_dropout, **block_kwargs,
        )
        self.tcn2 = EnhancedWaveletTCNBlock(
            c1, c2, wavelet=wavelet, dilation=2, dropout=block_dropout, **block_kwargs,
        )
        self.attention_weight = nn.Parameter(torch.ones(1, c2, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.tcn2(self.tcn1(self.input_conv(x)))) * self.attention_weight


class SimplifiedMultiWaveletFusion(nn.Module):

    def __init__(self, num_branches=6, channels=128, attention_reduction=4, dropout=0.3):
        super().__init__()
        self.num_branches = num_branches
        self.channels = channels
        self.branch_norms = nn.ModuleList([
            nn.InstanceNorm1d(channels, affine=True) for _ in range(num_branches)
        ])
        self.branch_weights = nn.Parameter(torch.zeros(num_branches))
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.ca = CoordAttention1D(channels, attention_reduction)
        self.dropout = nn.Dropout(dropout)

    def forward(self, branch_features):
        if len(branch_features) != self.num_branches:
            raise ValueError(f"Expected {self.num_branches} branch feature tensors")
        norm_feats = [norm(feat) for norm, feat in zip(self.branch_norms, branch_features)]
        weights = torch.softmax(self.branch_weights, dim=0)
        fused = sum(weight * feat for weight, feat in zip(weights, norm_feats))
        return self.dropout(self.ca(self.fusion_conv(fused))), weights


class MWAFNet(nn.Module):

    def __init__(
        self, num_classes=2, branch_channels=(64, 64, 128), post_channels=256,
        classifier_channels=128, se_reduction=4, attention_reduction=4,
        dropblock_max_prob=0.15, dropblock_warmup_epochs=15, dropblock_size=3,
        branch_dropout=0.2, block_dropout=0.2, fusion_dropout=0.3,
        classifier_dropout=0.2,
    ):
        super().__init__()
        if num_classes not in (2, 3, 5):
            raise ValueError("num_classes must be 2, 3 or 5")
        if len(branch_channels) != 3 or any(int(c) != c or c < 3 for c in branch_channels):
            raise ValueError("branch_channels must contain three integer widths >= 3")
        if post_channels < 3 or classifier_channels < 1:
            raise ValueError("post_channels must be >= 3 and classifier_channels >= 1")
        self.num_classes = num_classes
        self.config_version = MODEL_CONFIG_VERSION
        self.model_config = {
            "branch_channels": list(branch_channels),
            "post_channels": post_channels,
            "classifier_channels": classifier_channels,
            "se_reduction": se_reduction,
            "attention_reduction": attention_reduction,
            "dropblock_max_prob": dropblock_max_prob,
            "dropblock_warmup_epochs": dropblock_warmup_epochs,
            "dropblock_size": dropblock_size,
            "branch_dropout": branch_dropout,
            "block_dropout": block_dropout,
            "fusion_dropout": fusion_dropout,
            "classifier_dropout": classifier_dropout,
        }
        self.register_buffer("training_epoch", torch.tensor(0, dtype=torch.long))
        self.wavelets = ["morl", "db4", "sym4", "coif4", "haar", "mexh"]
        self.branch_out_channels = branch_channels[-1]
        block_kwargs = {
            "se_reduction": se_reduction,
            "attention_reduction": attention_reduction,
            "dropblock_prob": dropblock_max_prob,
            "dropblock_warmup_epochs": dropblock_warmup_epochs,
            "dropblock_size": dropblock_size,
        }
        self.branches = nn.ModuleList([
            WaveletBranch(
                wavelet, branch_channels=branch_channels, dropout=branch_dropout,
                block_dropout=block_dropout, **block_kwargs,
            ) for wavelet in self.wavelets
        ])
        self.fusion = SimplifiedMultiWaveletFusion(
            num_branches=len(self.wavelets), channels=self.branch_out_channels,
            attention_reduction=attention_reduction, dropout=fusion_dropout,
        )
        self.enhance = EnhancedWaveletTCNBlock(
            self.branch_out_channels, post_channels, wavelet="morl", dilation=4,
            dropout=block_dropout, **block_kwargs,
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(post_channels, classifier_channels),
            nn.BatchNorm1d(classifier_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_channels, num_classes),
        )
        self._initialize_weights()

    def configuration(self):
        """Constructor kwargs excluding num_classes; JSON serializable."""
        return deepcopy(self.model_config)

    def set_epoch(self, epoch):
        if int(epoch) != epoch or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.training_epoch.fill_(int(epoch))
        for module in self.modules():
            if isinstance(module, DropBlock1D):
                module.set_epoch(epoch)
        return self

    def _initialize_weights(self):
        # Exclude the actual convolution groups, not their nonexistent children.
        # A global Conv1d initializer must not overwrite wavelet templates.
        wavelet_convs = {
            id(getattr(module, f"conv{i}"))
            for module in self.modules() if isinstance(module, MultiScaleWaveletConv1d)
            for i in (1, 2, 3)
        }
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                if id(module) not in wavelet_convs:
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.InstanceNorm1d)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def model_inference(self, x):
        branch_features = [branch(x) for branch in self.branches]
        fused_features, _ = self.fusion(branch_features)
        enhanced_features = self.enhance(fused_features)
        return self.classifier(self.global_pool(enhanced_features))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[2] < 8:
            raise ValueError("Expected [batch, 1, samples] heart-sound segments with >= 8 samples")
        return self.model_inference(x)

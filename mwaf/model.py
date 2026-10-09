"""Public main network; no comparator or component-variant models."""

from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F
from .wavelets import generate_wavelet_kernel


class DropBlock1D(nn.Module):
    """Archived temporal DropBlock implementation."""

    def __init__(self, drop_prob=0.1, block_size=3):
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        gamma = self.drop_prob / self.block_size**2
        mask = (torch.rand(x.shape[0], 1, x.shape[-1], device=x.device) < gamma).float()
        mask = F.max_pool1d(
            mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2
        )
        mask = 1 - mask
        mask = mask.expand_as(x)
        x = x * mask * (mask.numel() / mask.sum())
        return x


class MultiScaleWaveletConv1d(nn.Module):
    """Three convolution groups using the archived kernel-initialization routine. Scale/shift parameters are not used to regenerate kernels during forward passes."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        wavelet="morl",
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.wavelet = wavelet
        branch1_ch = out_channels // 3
        branch2_ch = out_channels // 3
        branch3_ch = out_channels - branch1_ch - branch2_ch
        self.scale1 = nn.Parameter(torch.ones(branch1_ch, in_channels, 1) * 1.0)
        self.scale2 = nn.Parameter(torch.ones(branch2_ch, in_channels, 1) * 2.0)
        self.scale3 = nn.Parameter(torch.ones(branch3_ch, in_channels, 1) * 4.0)
        self.shift1 = nn.Parameter(torch.zeros(branch1_ch, in_channels, 1))
        self.shift2 = nn.Parameter(torch.zeros(branch2_ch, in_channels, 1))
        self.shift3 = nn.Parameter(torch.zeros(branch3_ch, in_channels, 1))
        self.conv1 = nn.Conv1d(
            in_channels,
            branch1_ch,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=1,
            bias=bias,
        )
        self.conv2 = nn.Conv1d(
            in_channels,
            branch2_ch,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=1,
            bias=bias,
        )
        self.conv3 = nn.Conv1d(
            in_channels,
            branch3_ch,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=1,
            bias=bias,
        )
        self._init_wavelet_kernels()

    def _init_wavelet_kernels(self):
        """初始化多尺度小波核（支持指定小波基）"""
        with torch.no_grad():
            kernel1 = generate_wavelet_kernel(
                self.wavelet,
                self.kernel_size,
                self.scale1,
                self.shift1,
                device=self.scale1.device,
            )
            kernel2 = generate_wavelet_kernel(
                self.wavelet,
                self.kernel_size,
                self.scale2,
                self.shift2,
                device=self.scale2.device,
            )
            kernel3 = generate_wavelet_kernel(
                self.wavelet,
                self.kernel_size,
                self.scale3,
                self.shift3,
                device=self.scale3.device,
            )
            self.conv1.weight.data = kernel1
            self.conv2.weight.data = kernel2
            self.conv3.weight.data = kernel3

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out


class CoordAttention1D(nn.Module):
    """Archived one-dimensional coordinate attention."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(
            channels, channels // reduction, kernel_size=1, stride=1, padding=0
        )
        self.bn1 = nn.BatchNorm1d(channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc_x = nn.Conv1d(
            channels // reduction, channels, kernel_size=1, stride=1, padding=0
        )
        self.fc_c = nn.Conv1d(
            channels // reduction, channels, kernel_size=1, stride=1, padding=0
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, w = x.size()
        x_avg = self.avg_pool(x)
        x_avg = self.fc1(x_avg)
        x_avg = self.bn1(x_avg)
        x_avg = self.relu(x_avg)
        att_x = self.fc_x(x_avg)
        att_x = self.sigmoid(att_x)
        att_x = att_x.expand(-1, -1, w)
        att_c = self.fc_c(x_avg)
        att_c = self.sigmoid(att_c)
        att_c = att_c.expand(-1, -1, w)
        out = x * att_x * att_c
        return out


class EnhancedWaveletTCNBlock(nn.Module):
    """Archived two-layer wavelet temporal residual block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        wavelet="morl",
        kernel_size=3,
        dilation=1,
        dropout=0.1,
        dropblock_prob=0.05,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = MultiScaleWaveletConv1d(
            in_channels,
            out_channels,
            kernel_size,
            wavelet=wavelet,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels, momentum=0.95)
        self.relu = nn.ReLU(inplace=True)
        self.dropblock = DropBlock1D(drop_prob=dropblock_prob, block_size=3)
        self.conv2 = MultiScaleWaveletConv1d(
            out_channels,
            out_channels,
            kernel_size,
            wavelet=wavelet,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels, momentum=0.95)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels, momentum=0.95),
            )
        self.ca = CoordAttention1D(out_channels, reduction=8)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropblock(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out)
        out += residual
        out = self.relu(out)
        return out


class WaveletBranch(nn.Module):
    """One fixed-topology branch of the six-branch main network."""

    def __init__(self, wavelet, in_channels=1, out_channels=128, device="cuda"):
        super().__init__()
        self.wavelet = wavelet
        self.device = device
        self.input_conv = nn.Sequential(
            MultiScaleWaveletConv1d(
                in_channels, 32, kernel_size=7, wavelet=wavelet, stride=2, padding=3
            ),
            nn.BatchNorm1d(32, momentum=0.95),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
        ).to(device)
        self.tcn1 = EnhancedWaveletTCNBlock(
            32,
            64,
            wavelet=wavelet,
            kernel_size=3,
            dilation=1,
            dropout=0.2,
            dropblock_prob=0.1,
        ).to(device)
        self.tcn2 = EnhancedWaveletTCNBlock(
            64,
            out_channels,
            wavelet=wavelet,
            kernel_size=3,
            dilation=2,
            dropout=0.2,
            dropblock_prob=0.1,
        ).to(device)
        self.attention_weight = nn.Parameter(torch.ones(1, out_channels, 1)).to(device)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = self.dropout(x)
        x = x * self.attention_weight
        return x


class SimplifiedMultiWaveletFusion(nn.Module):
    """Instance-normalized branches, learned global softmax weights and refinement."""

    def __init__(self, num_branches=6, channels=128):
        super().__init__()
        self.num_branches = num_branches
        self.channels = channels
        self.branch_norms = nn.ModuleList(
            [nn.InstanceNorm1d(channels, affine=True) for _ in range(num_branches)]
        )
        self.branch_weights = nn.Parameter(torch.ones(num_branches) / num_branches)
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.ca = CoordAttention1D(channels, reduction=8)
        self.dropout = nn.Dropout(0.3)

    def forward(self, branch_features):
        norm_feats = [
            norm(feat) for norm, feat in zip(self.branch_norms, branch_features)
        ]
        weights = torch.softmax(self.branch_weights, dim=0)
        fused = sum((w * feat for w, feat in zip(weights, norm_feats)))
        fused = self.fusion_conv(fused)
        fused = self.ca(fused)
        fused = self.dropout(fused)
        return (fused, weights)


class MWAFNet(nn.Module):
    """Main optimized six-branch MWAF-Net. The archived global initializer is deliberately retained, including its overwrite of initial wavelet-convolution coefficients."""

    def __init__(self, num_classes=2):
        super().__init__()
        if num_classes not in (2, 3, 5):
            raise ValueError("num_classes must be 2, 3 or 5")
        args = SimpleNamespace(num_classes=num_classes)
        device = torch.device("cpu")
        self.num_classes = args.num_classes
        self.device = device
        self.wavelets = ["morl", "db4", "sym4", "coif4", "haar", "mexh"]
        self.branch_out_channels = 128
        self.branches = nn.ModuleList(
            [
                WaveletBranch(
                    wavelet, out_channels=self.branch_out_channels, device=device
                )
                for wavelet in self.wavelets
            ]
        )
        self.fusion = SimplifiedMultiWaveletFusion(
            num_branches=6, channels=self.branch_out_channels
        ).to(device)
        self.enhance = EnhancedWaveletTCNBlock(
            self.branch_out_channels,
            256,
            wavelet="morl",
            kernel_size=3,
            dilation=4,
            dropout=0.2,
            dropblock_prob=0.1,
        ).to(device)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, self.num_classes),
        ).to(device)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if not any(
                    [isinstance(p, MultiScaleWaveletConv1d) for p in m.children()]
                ):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                m.weight = nn.Parameter(m.weight.to(self.device))
                if m.bias is not None:
                    m.bias = nn.Parameter(m.bias.to(self.device))
            elif isinstance(m, (nn.BatchNorm1d, nn.InstanceNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                m.weight = nn.Parameter(m.weight.to(self.device))
                m.bias = nn.Parameter(m.bias.to(self.device))
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                m.weight = nn.Parameter(m.weight.to(self.device))
                m.bias = nn.Parameter(m.bias.to(self.device))

    def model_inference(self, x):
        """核心推理逻辑"""
        x = x.to(next(self.parameters()).device)
        branch_features = []
        for branch in self.branches:
            feat = branch(x)
            branch_features.append(feat)
        fused_features, branch_weights = self.fusion(branch_features)
        self.branch_weights = branch_weights
        enhanced_features = self.enhance(fused_features)
        pooled = self.global_pool(enhanced_features)
        out = self.classifier(pooled)
        return out

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[2] < 8:
            raise ValueError("Expected [batch, 1, samples] heart-sound segments")
        return self.model_inference(x)

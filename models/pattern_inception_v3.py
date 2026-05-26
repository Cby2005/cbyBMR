"""Paper-aligned BMR image-pattern branch: BayarConv followed by Inception-v3."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3


class BayarConv2d(nn.Module):
    """Constrained convolution whose center is -1 and neighbors sum to 1."""

    def __init__(self, channels=3, kernel_size=5):
        super(BayarConv2d, self).__init__()
        if kernel_size % 2 != 1:
            raise ValueError("BayarConv2d requires an odd kernel size.")
        self.weight = nn.Parameter(torch.empty(channels, channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        mask = torch.ones(kernel_size, kernel_size)
        center = kernel_size // 2
        mask[center, center] = 0.0
        constrained_center = torch.zeros(kernel_size, kernel_size)
        constrained_center[center, center] = -1.0
        self.register_buffer("neighbor_mask", mask.view(1, 1, kernel_size, kernel_size))
        self.register_buffer("fixed_center", constrained_center.view(1, 1, kernel_size, kernel_size))
        self.padding = kernel_size // 2

    def forward(self, image):
        neighbors = self.weight * self.neighbor_mask
        denominator = neighbors.sum(dim=(2, 3), keepdim=True)
        safe_denominator = torch.where(
            denominator.abs() < 1e-6,
            torch.full_like(denominator, 1e-6),
            denominator,
        )
        constrained_weight = neighbors / safe_denominator + self.fixed_center
        return F.conv2d(image, constrained_weight, bias=None, stride=1, padding=self.padding)


class PatternInceptionV3(nn.Module):
    """InceptionNet-V3 image-pattern analyzer stated in the BMR paper."""

    def __init__(self, num_classes=768):
        super(PatternInceptionV3, self).__init__()
        self.bayar = BayarConv2d(channels=3, kernel_size=5)
        self.backbone = inception_v3(
            weights=None,
            aux_logits=False,
            transform_input=False,
            num_classes=num_classes,
            init_weights=False,
        )

    def forward(self, image):
        return self.backbone(self.bayar(image))

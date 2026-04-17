# ## Modified code from pretrained CIFAR10 resnet https://github.com/chenyaofo/pytorch-cifar-models/blob/master/pytorch_cifar_models/resnet.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Improved BasicBlock with dropout and better structure."""

    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    """A block consisting of multiple BasicBlocks."""

    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate
        )

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropRate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class CifarClassifier(nn.Module):
    """
    WideResNet-style classifier for CIFAR-10.

    Args:
        depth: Total depth of network (e.g., 28, 34, 40)
        widen_factor: Width multiplier (e.g., 10 for WRN-28-10)
        num_score_channels: Number of score channels from diffusion model
        num_classes: Number of output classes
        dropRate: Dropout probability
    """

    def __init__(
        self,
        depth=28,
        widen_factor=10,
        num_score_channels=0,
        num_classes=10,
        dropRate=0.3,
        scores=None,
    ):
        super(CifarClassifier, self).__init__()

        # Calculate number of blocks per layer
        assert (depth - 4) % 6 == 0, "Depth should be 6n+4 (e.g., 28, 34, 40)"
        n = (depth - 4) // 6

        # Calculate channel dimensions
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        # Calculate input channels
        in_channels = 3 + num_score_channels
        self.num_score_channels = num_score_channels

        print(f"Building WideResNet-{depth}-{widen_factor}")
        print(f"  Input channels: {in_channels} (3 RGB + {num_score_channels} score)")
        print(f"  Blocks per layer: {n}")
        print(f"  Channel progression: {nChannels}")
        print(f"  Dropout rate: {dropRate}")
        print(
            f"  Estimated parameters: ~{self._count_parameters(depth, widen_factor):.1f}M"
        )
        
        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )

        # Three blocks with increasing width
        self.block1 = NetworkBlock(
            n, nChannels[0], nChannels[1], BasicBlock, 1, dropRate
        )
        self.block2 = NetworkBlock(
            n, nChannels[1], nChannels[2], BasicBlock, 2, dropRate
        )
        self.block3 = NetworkBlock(
            n, nChannels[2], nChannels[3], BasicBlock, 2, dropRate
        )

        # Final batch norm and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.out_features != num_classes:
                    nn.init.normal_(m.weight, mean=0.1, std=0.05)
                m.bias.data.zero_()

    def _count_parameters(self, depth, widen_factor):
        """Rough parameter count estimation."""
        n = (depth - 4) // 6
        base_channels = 16 * widen_factor
        num_channels = [16, base_channels, 2 * base_channels, 4 * base_channels]

        # Rough estimate: each block has ~9 * in_ch * out_ch parameters
        params = (self.num_score_channels + 3) * 16 * 9
        for i in range(1, len(num_channels)):
            params += num_channels[i - 1] * num_channels[i] * 9 * n
            params += num_channels[i] * num_channels[i] * 9 * n
        return params / 1e6

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

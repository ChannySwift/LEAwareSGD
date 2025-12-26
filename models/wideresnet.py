"""WideResNet implementation (https://arxiv.org/abs/1605.07146)."""
from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic ResNet block."""

    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.is_in_equal_out = (in_planes == out_planes)
        self.conv_shortcut = (not self.is_in_equal_out) and nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False) or None

    def forward(self, x):
        if not self.is_in_equal_out:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.is_in_equal_out:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        if not self.is_in_equal_out:
            return torch.add(self.conv_shortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    """Layer container for blocks."""

    def __init__(self,
                 nb_layers,
                 in_planes,
                 out_planes,
                 block,
                 stride,
                 drop_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers,
                                      stride, drop_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride,
                    drop_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(
                block(i == 0 and in_planes or out_planes, out_planes,
                      i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """WideResNet class."""

    def __init__(self, depth, num_classes, widen_factor=1, drop_rate=0.0, contrastive=''):
        super(WideResNet, self).__init__()
        n_channels = [64, 64 * widen_factor, 128 * widen_factor, 256 * widen_factor, 512 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, n_channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = self._make_layer(block, n_channels[0], n_channels[1], n, stride=1, drop_rate=drop_rate)
        self.block2 = self._make_layer(block, n_channels[1], n_channels[2], n, stride=2, drop_rate=drop_rate)
        self.block3 = self._make_layer(block, n_channels[2], n_channels[3], n, stride=2, drop_rate=drop_rate)
        self.block4 = self._make_layer(block, n_channels[3], n_channels[4], n, stride=2, drop_rate=drop_rate)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_channels[4], num_classes)

        if 'contrastive' in contrastive:
            self.pro_head = nn.Linear(n_channels[4], 128)
            self.contrastive = True
        else:
            self.contrastive = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, block, in_planes, out_planes, num_blocks, stride, drop_rate):
        layers = []
        layers.append(block(in_planes, out_planes, stride, drop_rate))
        for _ in range(1, num_blocks):
            layers.append(block(out_planes, out_planes, 1, drop_rate))
        return nn.Sequential(*layers)

    def get_proj(self, fea):
        z = self.pro_head(fea)
        z = F.normalize(z, dim=-1)
        return z

    def forward(self, x):
        end_points = {}

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        end_points['Embedding'] = out

        if self.contrastive:
            end_points['Projection'] = self.get_proj(out)

        out = self.fc(out)
        end_points['Predictions'] = F.softmax(out, dim=-1)

        return out, end_points

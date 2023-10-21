# ==============================================================================
# Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics
# Original: https://github.com/mkotha/WaveRNN
# Modified: Yi Zhao (zhaoyi[at]nii.ac.jp)
# All rights reserved.
# ==============================================================================
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math


class DownsamplingEncoder(nn.Module):
    """Down-sampling encoder."""

    def __init__(self, channels: int, layer_specs):
        """
        Args:
            channels - Feature dimension size of hidden layers
        """
        super().__init__()

        self.convs_wide = nn.ModuleList() # Conv
        self.convs_1x1  = nn.ModuleList() # SegFC
        self.skips = []                   # Residual
        self.layer_specs = layer_specs
        prev_channels: int = 1
        total_scale = 1
        pad_left = 0
        for stride, ksz, dilation_factor in layer_specs:
            # Conv - 2x output channel for GatedTanh
            conv_wide = nn.Conv1d(prev_channels, 2 * channels, ksz, stride=stride, dilation=dilation_factor)
            wsize = 2.967 / math.sqrt(ksz * prev_channels)
            conv_wide.weight.data.uniform_(-wsize, wsize)
            conv_wide.bias.data.zero_()
            self.convs_wide.append(conv_wide)
            # SegFC
            conv_1x1 = nn.Conv1d(channels, channels, 1)
            conv_1x1.bias.data.zero_()
            self.convs_1x1.append(conv_1x1)
            # (Sparse)Residual
            prev_channels = channels
            skip = (ksz - stride) * dilation_factor
            pad_left += total_scale * skip
            self.skips.append(skip)
            # whole encoder scale
            total_scale *= stride
        self.pad_left = pad_left
        self.total_scale = total_scale

        # PostNet :: (B, Feat=f, Frame=frm) -> (B, Feat=f, Frame=frm) - SegFC-ReLU-SegFC
        self.final_conv_0 = nn.Conv1d(channels, channels, 1)
        self.final_conv_0.bias.data.zero_()
        self.final_conv_1 = nn.Conv1d(channels, channels, 1)
        # We don't set the bias to 0 here because otherwise the initial model
        # would produce the 0 vector when the input is 0, and it will make
        # the vq layer unhappy.

    def forward(self, series: Tensor):
        """
        Args:
            series :: (B, T) - Input series, waveform or fo series
        Returns:
                   :: (B, Frame, Feat)
        """

        # Reshape :: (B, T) -> (B, Feat=1, T) - Unsqueeze feature dim
        x = series.unsqueeze(1)

        # down-sampling layers
        for i, stuff in enumerate(zip(self.convs_wide, self.convs_1x1, self.layer_specs, self.skips)):
            conv_wide, segfc, layer_spec, skip = stuff
            stride, _, _ = layer_spec
            # Conv
            x1 = conv_wide(x)
            # GatedTanh
            x1_a, x1_b = x1.split(x1.size(1) // 2, dim=1)
            x2 = torch.tanh(x1_a) * torch.sigmoid(x1_b)
            x3 = segfc(x2)
            # (Sparse)Residual
            if i == 0:
                x = x3
            else:
                x = x3 + x[:, :, skip:skip+x3.size(2)*stride].view(x.size(0), x3.size(1), x3.size(2), -1)[:, :, :, -1]

        # PostNet :: (B, Feat, Frame) -> (B, Feat, Frame) - SegFC-ReLU-SegFC
        x = self.final_conv_1(F.relu(self.final_conv_0(x)))

        # Reshpae :: (B, Feat, Frame) -> (B, Frame, Feat) - Transpose
        x = x.transpose(1, 2)

        return x

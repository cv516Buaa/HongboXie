# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from mmengine.model import BaseModule
import math
import numpy as np

@MODELS.register_module()
class HFDHead(BaseDecodeHead):
    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.AVGpool2 = nn.AvgPool2d(2, stride=2)
        self.AVGpool8 = nn.AvgPool2d(8, stride=8)

        self.alpha = 0.1
        self.Linear = nn.Conv2d(self.channels, 3 * self.channels, 1)
        self.lcd = 0
        self.mu = 0.05

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output_ = self.AVGpool8(output)
        b,c,H,W = output_.shape

        Low_output = self.AVGpool2(output_)
        Low_output = F.interpolate(Low_output, size=output_.shape[-2:], mode = 'bilinear')
        HF_map = torch.sum(output_ - Low_output,dim=1) # b h w
        one = torch.ones_like(HF_map) # b h w

        S = one - self.alpha * HF_map
        S[:,1:,:] += 0.25 * self.alpha * HF_map[:,:-1,:]
        S[:,:-1,:] += 0.25 * self.alpha * HF_map[:,1:,:]
        S[:,:,1:] += 0.25 * self.alpha * HF_map[:,:,:-1]
        S[:,:,:-1] += 0.25 * self.alpha * HF_map[:,:,1:]

        # F3 = output.view(b,c,H*W)
        F3 = self.Linear(output_)
        F3 = F3.view(b, 3 * c, H * W)
        q, k, v = F3[:,:c,:], F3[:,c:2*c,:], F3[:,2*c:,:]
        # print(q.shape)
        attn = torch.sum(q.unsqueeze(2) * k.unsqueeze(-1), 1) / math.sqrt(c)
        attn = F.softmax(attn,dim=-1)
        S = S.view(b, -1)
        # print(S.shape)
        attn_new = attn * S.unsqueeze(-2)
        v_new = torch.sum(attn_new.unsqueeze(1) * v.unsqueeze(-1), 2)
        v_new = v_new.view(b,c,H,W)
        v_new = F.interpolate(v_new, size=output.shape[-2:], mode = 'bilinear')
        output = (1 - self.mu) * output + self.mu * v_new
        output = self.cls_seg(output)
        return output

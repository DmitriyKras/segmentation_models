#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 22:20:16 2018

@author: Taha Emara  @email: taha@emaraic.com
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from collections import OrderedDict


def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1,dilation=1):    
    """3x3 convolution with padding
    """
    return nn.Conv2d(
        in_channels, 
        out_channels, 
        kernel_size=3, 
        stride=stride,
        padding=padding,
        bias=bias,dilation=dilation,
        groups=groups)


def conv1x1(in_channels, out_channels, groups=1):
    """1x1 convolution with padding
    - Normal pointwise convolution When groups == 1
    - Grouped pointwise convolution when groups > 1
    """
    return nn.Conv2d(
        in_channels, 
        out_channels, 
        kernel_size=1, 
        groups=groups,
        stride=1)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3,
                 grouped_conv=True, combine='add',dilation=1):
        
        super(ShuffleUnit, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grouped_conv = grouped_conv
        self.combine = combine
        self.groups = groups
        self.dilation=dilation
        self.bottleneck_channels = self.out_channels // 4

        # define the type of ShuffleUnit
        if self.combine == 'add':
            # ShuffleUnit Figure 2b
            self.depthwise_stride = 1
            self._combine_func = self._add
        elif self.combine == 'concat':
            # ShuffleUnit Figure 2c
            self.depthwise_stride = 2
            self._combine_func = self._concat
            
            # ensure output of concat has the same channels as 
            # original output channels.
            self.out_channels -= self.in_channels
        else:
            raise ValueError("Cannot combine tensors with \"{}\"" \
                             "Only \"add\" and \"concat\" are" \
                             "supported".format(self.combine))

        # Use a 1x1 grouped or non-grouped convolution to reduce input channels
        # to bottleneck channels, as in a ResNet bottleneck module.
        # NOTE: Do not use group convolution for the first conv1x1 in Stage 2.
        self.first_1x1_groups = self.groups if grouped_conv else 1

        self.g_conv_1x1_compress = self._make_grouped_conv1x1(
            self.in_channels,
            self.bottleneck_channels,
            self.first_1x1_groups,
            batch_norm=True,
            relu=True
            )

        # 3x3 depthwise convolution followed by batch normalization
        self.depthwise_conv3x3 = conv3x3(
            self.bottleneck_channels, self.bottleneck_channels,
            stride=self.depthwise_stride, groups=self.bottleneck_channels,dilation=self.dilation)
        self.bn_after_depthwise = nn.BatchNorm2d(self.bottleneck_channels)

        # Use 1x1 grouped convolution to expand from 
        # bottleneck_channels to out_channels
        self.g_conv_1x1_expand = self._make_grouped_conv1x1(
            self.bottleneck_channels,
            self.out_channels,
            self.groups,
            batch_norm=True,
            relu=False
            )


    @staticmethod
    def _add(x, out):
        # residual connection
        return x + out


    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)


    def _make_grouped_conv1x1(self, in_channels, out_channels, groups,
        batch_norm=True, relu=False):

        modules = OrderedDict()

        conv = conv1x1(in_channels, out_channels, groups=groups)
        modules['conv1x1'] = conv

        if batch_norm:
            modules['batch_norm'] = nn.BatchNorm2d(out_channels)
        if relu:
            modules['relu'] = nn.ReLU()
        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv


    def forward(self, x):
        # save for combining later with output
        residual = x

        if self.combine == 'concat':
            residual = F.avg_pool2d(residual, kernel_size=3, 
                stride=2, padding=1)

        out = self.g_conv_1x1_compress(x)
        out = channel_shuffle(out, self.groups)
        out = self.depthwise_conv3x3(out)
        out = self.bn_after_depthwise(out)
        out = self.g_conv_1x1_expand(out)
        
        out = self._combine_func(residual, out)
        return F.relu(out)


class ShuffleNet(nn.Module):
    """ShuffleNet implementation.
    """

    def __init__(self, groups=8, in_channels=3, num_classes=1000):
        """ShuffleNet constructor.

        Arguments:
            groups (int, optional): number of groups to be used in grouped 
                1x1 convolutions in each ShuffleUnit. Default is 3 for best
                performance according to original paper.
            in_channels (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
            num_classes (int, optional): number of classes to predict. Default
                is 1000 for ImageNet.

        """
        super(ShuffleNet, self).__init__()

        self.groups = groups
        self.stage_repeats = [3, 7, 3]
        self.in_channels =  in_channels
        self.num_classes = num_classes

        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if groups == 1:
            self.stage_out_channels = [-1, 24, 144, 288, 567]
        elif groups == 2:
            self.stage_out_channels = [-1, 24, 200, 400, 800]
        elif groups == 3:
            self.stage_out_channels = [-1, 24, 240, 480, 960]
        elif groups == 4:
            self.stage_out_channels = [-1, 24, 272, 544, 1088]
        elif groups == 8:
            self.stage_out_channels = [-1, 24, 384, 768, 1536]
        else:
            raise ValueError(
                """{} groups is not supported for
                   1x1 Grouped Convolutions""".format(groups))
        
        # Stage 1 always has 24 output channels
        self.conv1 = conv3x3(self.in_channels,
                             self.stage_out_channels[1], # stage 1
                             stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 2
        self.stage2 = self._make_stage(2)
        # Stage 3
        self.stage3 = self._make_stage(3)
        # Stage 4
        self.stage4 = self._make_stage(4,dilation=2)
        
        num_inputs = self.stage_out_channels[-1]
        self.fc = nn.Linear(num_inputs, self.num_classes)

    

    def _make_stage(self, stage,dilation=1):
        modules = OrderedDict()
        stage_name = "ShuffleUnit_Stage{}".format(stage)
        
        # First ShuffleUnit in the stage
        # 1. non-grouped 1x1 convolution (i.e. pointwise convolution)
        #   is used in Stage 2. Group convolutions used everywhere else.
        grouped_conv = stage > 2
        
        # 2. concatenation unit is always used.
        first_module = ShuffleUnit(
            self.stage_out_channels[stage-1],
            self.stage_out_channels[stage],
            groups=self.groups,
            grouped_conv=grouped_conv,
            combine='concat'#,dilation=dilation
            )
        modules[stage_name+"_0"] = first_module

        # add more ShuffleUnits depending on pre-defined number of repeats
        for i in range(self.stage_repeats[stage-2]):
            name = stage_name + "_{}".format(i+1)
            module = ShuffleUnit(
                self.stage_out_channels[stage],
                self.stage_out_channels[stage],
                groups=self.groups,
                grouped_conv=True,
                combine='add'#,dilation=dilation
                )
            modules[name] = module

        return nn.Sequential(modules)


    def forward(self, x):
        #print('input to network',x.size())
        x = self.conv1(x)
        x = self.maxpool(x)
        #print('Shuffle after conv1',x.size())
        
        x = self.stage2(x)
        #print('after stage 2',x.size())

        low_features=x
        x = self.stage3(x)
        #print('after stage 3',x.size())

        x = self.stage4(x)
        #print('after stage 4',x.size())


#       # global average pooling layer
#        x = F.avg_pool2d(x, x.data.size()[-2:])
#        
#        # flatten for input to fully-connected layer
#        x = x.view(x.size(0), -1)
#        x = self.fc(x)
#
#        return F.log_softmax(x, dim=1)
        
        return x,low_features


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
class ASPP(nn.Module):
    
    def __init__(self, inplanes, planes, rate):
        super(ASPP, self).__init__()
        self.rate=rate
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
            #self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, bias=False,padding=1)
            self.conv1 =SeparableConv2d(planes,planes,3,1,1)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu1 = nn.ReLU()
   
            #self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
            #                         stride=1, padding=padding, dilation=rate, bias=False)
        self.atrous_convolution = SeparableConv2d(inplanes,planes,kernel_size,1,padding,rate)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        
        
        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        #x = self.relu(x)
        if self.rate!=1:
            x=self.conv1(x)
            x=self.bn1(x)
            x=self.relu1(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class LiteSegShuffleNet(nn.Module):
    
    def __init__(self, n_classes=19,pretrained=True,PRETRAINED_WEIGHTS="."):

        super(LiteSegShuffleNet, self).__init__()
        print("RTShufflenet model...")

        self.shuffle_features=ShuffleNet()
        self.shuffle_features = torch.nn.DataParallel(self.shuffle_features)
        if pretrained:
            state_dict = torch.load(PRETRAINED_WEIGHTS)
            self.shuffle_features.load_state_dict(state_dict)
        
       
        rates =[1, 3, 6, 9]

        self.aspp1 = ASPP(1536, 96, rate=rates[0])
        self.aspp2 = ASPP(1536, 96, rate=rates[1])
        self.aspp3 = ASPP(1536, 96, rate=rates[2])
        self.aspp4 = ASPP(1536, 96, rate=rates[3])

        self.relu = nn.ReLU()
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(1536, 96, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(96),
                                             nn.ReLU())
        #self.conv1 = nn.Conv2d(480+1536, 96, 1, bias=False)#+1536
        self.conv1 =SeparableConv2d(480+1536,96,1)
        self.bn1 = nn.BatchNorm2d(96)

        # adopt [1x1, 48] for channel reduction.
        #self.conv2 = nn.Conv2d(384, 48, 1, bias=False)
        self.conv2 = SeparableConv2d(384,48,1)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(#nn.Conv2d(48+96, 96, kernel_size=3, stride=1, padding=1, bias=False),
                                       SeparableConv2d(48+96,96,3,1,1),
                                       nn.BatchNorm2d(96),
                                       nn.ReLU(),
                                       #nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False),
                                       SeparableConv2d(96,96,3,1,1),
                                       nn.BatchNorm2d(96),
                                       nn.ReLU(),
                                       nn.Conv2d(96, n_classes, kernel_size=1, stride=1))
        self.activation = nn.Sigmoid()
        
        
    def forward(self, input):
        x, low_level_features = self.shuffle_features(input)
        #print(x.size())
        x1 = self.aspp1(x)
        #print("after aspp 1",x1.size())
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x,x1, x2, x3, x4, x5), dim=1)
        #print('after aspp cat',x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2]/8)),
                                int(math.ceil(input.size()[-1]/8))), mode='bilinear', align_corners=True)
        #print('after con on aspp output',x.size())

        ##comment to remove low feature
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)
        #print("low",low_level_features.size())
        x = torch.cat((x, low_level_features), dim=1)
        #print('after cat low feature with output of aspp',x.size())
        ##   
        
        x = self.last_conv(x)
        #print(x.size())
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return self.activation(x)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

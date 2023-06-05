
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# refer : https://github.com/Tianxiaomo/pytorch-YOLOv4
class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, depthwise=False, dilation=1, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if depthwise and dilation == 1:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, dilation=dilation, groups=in_channels, bias=bias))
        elif dilation == 1:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, dilation=dilation, bias=bias))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation, bias=bias))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                       sys._getframe().f_code.co_name, sys._getframe().f_lineno))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class DualPathBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, dilation):
        super().__init__()

        self.conv1 = nn.Sequential(
            Conv_Bn_Activation(inplanes, inplanes//4, 1, 1, 'relu'),
            Conv_Bn_Activation(inplanes//4, inplanes//2, 3, 1, 'linear')
        )

        self.conv2 = nn.Sequential(
            Conv_Bn_Activation(inplanes, inplanes//2, 3, 1, 'relu', dilation=dilation),
            Conv_Bn_Activation(inplanes//2, inplanes//2, 3, 1, 'linear', dilation=dilation)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = torch.cat((out1, out2), dim=1) + x
        return self.relu(out3)


class FeatureExtractionNetwork_dualpath(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = Conv_Bn_Activation(5, 64, 7, 2, 'relu') #1/2
        self.conv2 = Conv_Bn_Activation(64, 128, 5, 2, 'relu') #1/4

        self.block1 = DualPathBlock(128, 2)
        self.block2 = DualPathBlock(128, 2)
        self.block3 = DualPathBlock(128, 2)
        self.block4 = DualPathBlock(128, 2)
        self.block5 = DualPathBlock(128, 2)
        self.block6 = DualPathBlock(128, 2)
        self.block7 = DualPathBlock(128, 2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        b1 = self.block1(x2)
        b2 = self.block2(b1)
        b3 = self.block3(b2)
        b4 = self.block4(b3)
        b5 = self.block5(b4)
        b6 = self.block6(b5)
        b7 = self.block7(b6)
        return b7

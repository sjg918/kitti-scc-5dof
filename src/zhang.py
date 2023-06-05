from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses import quaternion_distance


# refer : https://github.com/JiaRenChang/PSMNet

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


class BasicBlock_(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock_, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class Zhang_ResNet(nn.Module):

    def __init__(self):
        self.inplanes = 64

        super(Zhang_ResNet, self).__init__()

        self.firstconv = nn.Sequential(convbn(5, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 64, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True))

        # 1 1 2 4 1 1
        self.layer1 = BasicBlock_(64, 64, 1, None, 1, 1)
        self.layer2 = BasicBlock_(64, 64, 1, None, 1, 1)
        self.layer3 = BasicBlock_(64, 64, 1, None, 1, 2)
        self.layer4 = BasicBlock_(64, 64, 1, None, 1, 4)
        self.layer5 = BasicBlock_(64, 64, 1, None, 1, 1)
        self.layer6 = BasicBlock_(64, 64, 1, None, 1, 1)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        return x



# refer : https://github.com/JiaRenChang/PSMNet

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1, dilation=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1, dilation=1)

        self.conv3 = nn.Sequential(convbn(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1, dilation=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1, dilation=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose2d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm2d(inplanes*2)) #+conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose2d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm2d(inplanes)) #+x

    def forward(self, x ,presqu, postsqu):
        
        out  = self.conv1(x) #in:1/4 out:1/8
        pre  = self.conv2(out) #in:1/8 out:1/8
        if postsqu is not None:
           pre = F.relu(pre + postsqu, inplace=True)
        else:
           pre = F.relu(pre, inplace=True)

        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16

        if presqu is not None:
           post = F.relu(self.conv5(out)+presqu, inplace=True) #in:1/16 out:1/8
        else:
           post = F.relu(self.conv5(out)+pre, inplace=True) 

        out  = self.conv6(post)  #in:1/8 out:1/4

        return out

class Zhang_base(nn.Module):
    def __init__(self):
        super(Zhang_base, self).__init__()
        self.dres0 = nn.Sequential(convbn(128, 64, 3, 1, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn(64, 64, 3, 1, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn(64, 64, 3, 1, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn(64, 64, 3, 1, 1, 1)) 

        self.dres2 = hourglass(64)

        self.dres3 = hourglass(64)

        self.dres4 = hourglass(64)

        self.out1 = nn.Sequential(convbn(64, 128, 3, 2, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn(128, 256, 3, 2, 1, 1),
                                   nn.ReLU(inplace=True))

        self.fc_left_rot = nn.Linear(243712, 4)
        self.fc_right_rot = nn.Linear(243712, 4)

    def forward(self, left, right, left_target, right_target):
        B, C, H, W = left.shape

        x = self.dres0(torch.cat((left, right), dim=1))
        x = self.dres1(x) + x
        x = self.dres2(x, None, None) + x
        x = self.dres3(x, None, None) + x
        x = self.dres4(x, None, None) + x
        x = self.out1(x)
        
        x = x.view(B, -1)
        
        leftout = self.fc_left_rot(x)
        rightout = self.fc_right_rot(x)
        left_quaternion = F.normalize(leftout, dim=1)
        right_quaternion = F.normalize(rightout, dim=1)

        if left_target is not None:
            left_rotloss = quaternion_distance(left_quaternion, left_target).mean()
            right_rotloss = quaternion_distance(right_quaternion, right_target).mean()
            return left_rotloss, right_rotloss
        return left_quaternion, right_quaternion
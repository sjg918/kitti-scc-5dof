
import torch
import torch.nn as nn

import torch.nn.functional as F
from src.losses import quaternion_distance


def myconv(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU())


class CrossAttentionNet_base(nn.Module):
    def __init__(self, in_channels, qkv_channels):
        super().__init__()
        self.qkv_channels = qkv_channels

        # pvcn
        self.patch_left = nn.Conv2d(in_channels, qkv_channels, kernel_size=4, stride=4, padding=0, bias=False)
        self.layernorm_left = nn.LayerNorm(qkv_channels)
        self.patch_right = nn.Conv2d(in_channels, qkv_channels, kernel_size=4, stride=4, padding=0, bias=False)
        self.layernorm_right = nn.LayerNorm(qkv_channels)
        self.relu = nn.ReLU(inplace=True)

        # pwcam
        self.match_q = nn.Sequential(
            nn.Linear(qkv_channels, qkv_channels, bias=False),
            nn.LayerNorm(qkv_channels),
            nn.ReLU(inplace=True)
        )
        self.match_k = nn.Sequential(
            nn.Linear(qkv_channels, qkv_channels, bias=False),
            nn.LayerNorm(qkv_channels),
            nn.ReLU(inplace=True)
        )

        # rn
        self.conv1 = myconv(qkv_channels*4, qkv_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = myconv(qkv_channels, qkv_channels*2, kernel_size=3, stride=2, padding=1)
        self.conv3 = myconv(qkv_channels*2, qkv_channels*4, kernel_size=3, stride=2, padding=1)
        self.conv4 = myconv(qkv_channels*4, qkv_channels*8, kernel_size=3, stride=2, padding=1)
        self.left_regress = nn.Linear(18 * qkv_channels*8, 4, bias=True)
        self.right_regress = nn.Linear(18 * qkv_channels*8, 4, bias=True)


    def match(self, x, y):
        x = self.patch_left(x)
        y = self.patch_right(y)
        B, C, H, W = x.shape

        x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()
        y = y.view(B, C, H*W).permute(0, 2, 1).contiguous()
        x = self.relu(self.layernorm_left(x))
        y = self.relu(self.layernorm_right(y))

        q = self.match_q(x)
        k = self.match_k(y)

        attn1 = q @ k.permute(0, 2, 1).contiguous()
        attn1 = F.softmax(attn1, dim=-1)
        attn1 = attn1 @ y
        attn1 = torch.cat((attn1, x), dim=-1)

        attn2 = k @ q.permute(0, 2, 1).contiguous()
        attn2 = F.softmax(attn2, dim=-1)
        attn2 = attn2 @ x
        attn2 = torch.cat((attn2, y), dim=-1)

        attn = torch.cat((attn1, attn2), dim=1)
        attn = attn.permute(0, 2, 1).contiguous().view(B, C*4, H, W)
        return attn


    def forward(self, left, right, left_target, right_target):
        B, C, H, W = left.shape

        # pvcn + pwcam
        attnvol0 = self.match(left, right)

        # rn
        attnvol0 = self.conv1(attnvol0)
        attnvol1 = self.conv2(attnvol0)
        attnvol7 = self.conv3(attnvol1)
        attnvol13 = self.conv4(attnvol7)
        attnvol13 = attnvol13.view(B, -1)
        leftout = self.left_regress(attnvol13)
        rightout = self.right_regress(attnvol13)
        left_quaternion = F.normalize(leftout, dim=1)
        right_quaternion = F.normalize(rightout, dim=1)

        # loss
        if left_target is not None:
            left_rotloss = quaternion_distance(left_quaternion, left_target).mean()
            right_rotloss = quaternion_distance(right_quaternion, right_target).mean()
            return left_rotloss, right_rotloss
        return left_quaternion, right_quaternion
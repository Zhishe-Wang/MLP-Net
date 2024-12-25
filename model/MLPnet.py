import os

import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
from torchsummary import summary
from thop import profile, clever_format
from ast import Pass
from email.quoprimime import body_check

# from turtle import color, forward
from matplotlib.pyplot import axis
from numpy import append

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

from timm.models.layers import to_2tuple, trunc_normal_, DropPath




class CSFM(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_atten = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_redu = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(embed_dim, 1, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(embed_dim, 1, kernel_size=1, stride=1, bias=True)
        self.Upsample = UpsampleBlock(embed_dim * 2, embed_dim)
        self.nonlin = nn.Sigmoid()

    def forward(self, x, skip):
        skip = self.Upsample(skip)
        output = torch.cat([x, skip], dim=1)

        att = self.conv_atten(self.avg_pool(output))
        output = output * att
        output = self.conv_redu(output)

        att = self.conv1(x) + self.conv2(skip)
        att = self.nonlin(att)
        output = output * att
        return output


########################################################################################################################

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 256.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 80.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=256, patch_size=4, in_chans=3, embed_dim=80, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:

            x = self.norm(x)
            print(x.shape)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops

class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 256.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 80.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=256, patch_size=4, in_chans=3, embed_dim=80, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

#################################################################################################################################
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        # 步长为2的2x2转置卷积
        self.transposed_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0
        )
        # 批量归一化
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        # GeLU 激活函数
        self.gelu1 = nn.GELU()
        # 步长为1的3x3卷积
        self.conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        # 另一个批量归一化
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        # 另一个 GeLU 激活函数
        self.gelu2 = nn.GELU()

    def forward(self, x):
        x = self.transposed_conv(x)
        x = self.batch_norm1(x)
        x = self.gelu1(x)
        x = self.conv(x)
        x = self.batch_norm2(x)
        x = self.gelu2(x)
        return x

class _FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.block(x)

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.conv = ConvLayer(4 * dim, 2 * dim, 1, 1)


    def forward(self, x):
        B, C, H, W = x.shape

        # 确保H和W都是偶数
        if H % 2 != 0 or W % 2 != 0:
            print(f"Warning: x.shape {x.shape} is not divisible by 2 evenly. Padding might be needed.", flush=True)
            # 这里你可以选择进行填充或者裁剪，但为了简单起见，我们只裁剪
            H_new, W_new = H // 2, W // 2
            x = x[:, :H_new, :W_new, :]

            # 分割并合并patch
        x0 = x[:, :, 0::2, 0::2]  # B H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2]  # B H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2]  # B H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2]  # B H/2 W/2 C

        x = torch.cat([x0, x1, x2, x3], dim=-1)  # B H/2 W/2 4*C
        x = x.view(B, 4 * C, H // 2, W // 2)  # B H/2 W/2 4*C
        x = self.conv(x)



        # 应用变换和归一化
        # x = self.norm(x)
        # x = self.reduction(x)  # B H/2 W/2 2*C


        return x

##################################################################################################################################

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )


    def forward(self, x):
        residual = x
        x = self.body(x)


        out = F.relu(x+residual, True)
        return out


class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.ReLU1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.ReLU1(self.fc1(self.avg_pool(x))))  #1,1，c
        max_out = self.fc2(self.ReLU1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out   #1,1，c
        return self.sigmoid(out)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Res_CBAM_block(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1,):
        super(Res_CBAM_block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU(inplace=True)  # in relu inplace = true
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()
        # self.rcb = ResidualBlock(out_channels, out_channels)

        # self.dropout = nn.Dropout(0.3) # Dropout

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.ReLU(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out   #h,w,c
        out = self.sa(out) * out
        out += residual
        out = self.ReLU(out)
        return out


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.is_last = is_last
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu= nn.PReLU()
    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)

        if self.is_last is False:
            out = self.prelu(out)
        return out


##############################################################################################################################



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):  # 嵌入向量维度，相对位置编码中允许的最大相对距离
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)
        trunc_normal_(self.embeddings_table, std=.02)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat)
        embeddings = self.embeddings_table[final_mat]

        return embeddings


class PTIM(nn.Module):
    def __init__(self, channels, H, W, alpha, use_dropout=False, drop_rate=0):
        super().__init__()
        assert W == H
        self.channels = channels
        self.activation = nn.GELU()
        self.BN = nn.BatchNorm2d(channels // 2)

        if channels % 64 == 0:
            patch = 2
        else:
            patch = 4  # 如果通道是64的整数倍，patch=2，其他patch=4

        self.ratio = 1;
        self.C = int(channels * 0.5 / patch);

        self.chan = self.ratio * self.C
        # 这行代码首先将channels乘以0.5，然后除以patch，得到的结果再通过int()函数转换为整数，结果被赋值给类的C属性。
        self.proj_h = nn.Conv2d(H * self.C, self.chan * H, (1, 3), stride=1, padding=(0, 1), groups=self.C, bias=True)
        self.proj_w = nn.Conv2d(self.C * W, self.chan * W, (1, 3), stride=1, padding=(0, 1), groups=self.C, bias=True)

        self.fuse_h = nn.Conv2d(channels, channels // 2, (1, 1), (1, 1), bias=False)
        self.fuse_w = nn.Conv2d(channels, channels // 2, (1, 1), (1, 1), bias=False)

        self.mlp = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, bias=True), nn.BatchNorm2d(channels), nn.GELU())

        dim = channels // 2
        # 对x2的操作
        self.fc_h = nn.Conv2d(dim, dim, (3, 7), stride=1, padding=(1, 7 // 2), groups=dim,
                              bias=False)  # 这意味着卷积核的高度是3，宽度是7，表明这个卷积层可能专注于处理水平的空间模式。
        self.fc_w = nn.Conv2d(dim, dim, (7, 3), stride=1, padding=(7 // 2, 1), groups=dim,
                              bias=False)  # 这意味着卷积核的高度是7，宽度是3，表明这个卷积层可能专注于处理垂直的空间模式。

        self.reweight = Mlp(dim, dim // 2, dim * 3)  # 输入层维度，中间层，输出层维度

        self.fuse = nn.Conv2d(channels, channels, (1, 1), (1, 1), bias=False)
        self.fuse1 = nn.Conv2d(channels * 2, channels, (1, 1), (1, 1), bias=False)
        self.relate_pos_h = RelativePosition(channels // 2, H)
        self.relate_pos_w = RelativePosition(channels // 2, W)

        self.MLPC = MLPC(channels // 2, alpha, use_dropout, drop_rate)
        drop_rate = 0.1
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

        self.Pc = Partial_conv3(channels)

        self.pw = nn.Sequential(
        nn.Conv2d(channels, channels, 1, 1, 0),
        nn.BatchNorm2d(channels),
        nn.ReLU(True),  # PW Conv

        nn.Conv2d(channels, channels, 1, 1, 0),
        nn.BatchNorm2d(channels),
        )


    def forward(self, x):
        N, C, H, W = x.shape  # 将x分为四个维度NCHW

        x = self.mlp(x)  # 学习特征间复杂的非线性关系。这里self.mlp对输入进行了变换，学习了更高层的特征表示。PS.就是一个1x1卷积没有改变通道数。
        x1 = self.Pc(x)
        x2 = self.pw(x1)


        x_1 = x2[:, :C // 2, :, :]
        x_2 = x2[:, C // 2:, :, :]
        # 将x沿通道维度分成两部分
        x_1 = self.MLP_h(x_1)
        x_2 = self.MLP_w(x_2)


        x = self.fuse(torch.cat([x_1, x_2], dim=1)) + x  # cat起来

        # #print(x.shape)
        return x

    def MLP_h(self, x):
        N, C, H, W = x.shape    #8 40 64 64

        pos_h = self.relate_pos_h(H, W).unsqueeze(0).permute(0, 3, 1, 2)   #1 40 64 64

        # pos_w = self.relate_pos_w(H, W).unsqueeze(0).permute(0, 3, 1, 2)  #1 40 64 64

        C1 = int(C / self.C)

        x_h = x + pos_h  #8 40 64 64

        x_h = x_h.view(N, C1, self.C, H, W)  # N C1 C2 H W     8 40 64 64   --  8 2 20 64 64原来的每个通道现在被分组为小组，每组有8个通道，总共有8组。

        x_h = x_h.permute(0, 1, 3, 2, 4).contiguous().view(N, C1, H, self.C * W)  # N C1 H WC2   8 2 64 1280

        x_h = self.proj_h(x_h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)   # 1x3的卷积处理 N C1 H WC2  后再排回N C H W    8 2 64 1280
        x_h = self.proj_h(x_h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) + x_h  # 1x3的卷积处理 N C1 H WC2  后再排回N C H W    8 2 64 1280

        x_h = x_h.view(N, C1, H, self.C, W).permute(0, 1, 3, 2, 4).contiguous().view(N, C, H, W)  # N C1 C2 H W
        #print(x_h.shape)
        x = self.fuse_h(torch.cat([x_h, x], dim=1))
        x = self.activation(self.BN(x))



        x = x + self.drop_path(self.MLPC(x))






        return x

    def MLP_w(self, x):
        N, C, H, W = x.shape  # 8 40 64 64


        pos_w = self.relate_pos_w(H, W).unsqueeze(0).permute(0, 3, 1, 2)  # 1 40 64 64

        C1 = int(C / self.C)

        x_w = x + pos_w  # 8 40 64 64

        x_w = x_w.view(N, C1, self.C, H,
                       W)  # N C1 C2 H W     8 40 64 64   --  8 2 20 64 64原来的每个通道现在被分组为小组，每组有8个通道，总共有8组。


        x_w = x_w.permute(0, 1, 3, 2, 4).contiguous().view(N, C1, self.C * H,  W)  # N C1 H WC2   8 2 64 1280

        x_w = self.proj_w(x_w.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)  # 1x3的卷积处理 N C1 H WC2  后再排回N C H W    8 2 64 1280
        x_w = self.proj_w(x_w.permute(0, 2, 1, 3)).permute(0, 2, 1, 3) + x_w # 1x3的卷积处理 N C1 H WC2  后再排回N C H W    8 2 64 1280

        x_w = x_w.view(N, C1, H, self.C, W).permute(0, 1, 3, 2, 4).contiguous().view(N, C, H, W)  # N C1 C2 H W

        x_w = self.fuse_h(torch.cat([x_w, x], dim=1))
        x = self.activation(self.BN(x_w))

        x = x + self.drop_path(self.MLPC(x))


        return x




class TokenMixing(nn.Module):
    r""" Token mixing of Strip MLP

    Args:
        img_size (int): Image size.  Default: 256.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, C, H, W, alpha, use_dropout=False, drop_rate=0):
        super().__init__()
        self.PTIM_block = PTIM(C, H, W, alpha, use_dropout=False, drop_rate=0)



    def forward(self, x):

        x = self.PTIM_block(x) + x

        return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class MLPC(nn.Module):

    def __init__(self, in_channel, alpha, use_dropout=False, drop_rate=0):
        super().__init__()

        self.use_dropout = use_dropout

        self.conv_77 = nn.Conv2d(in_channel, in_channel, 7, 1, 3, groups=in_channel, bias=False)
        self.layer_norm = nn.LayerNorm(in_channel)
        self.fc1 = nn.Linear(in_channel, alpha * in_channel)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(alpha * in_channel, in_channel)

        self.grn = GRN(3 * in_channel)

    def forward(self, x):
        N, C, H, W = x.shape
        x = self.conv_77(x)
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.grn(x)

        x = self.fc2(x)

        x = x.permute(0, 3, 1, 2)

        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channel, H, W, alpha, use_dropout=False, drop_rate=0):
        super().__init__()

        self.T_Cmix = TokenMixing(in_channel, H, W, alpha, use_dropout=False, drop_rate=0)

        drop_rate = 0.1

        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.T_Cmix(x))

        return x

class MLPNet(nn.Module):
    """
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        layers (tuple(int)): Depth of each Swin Transformer layer. 2,8,14,2
        drop_rate (float): Dropout rate. Default: 0
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
    """

    def __init__(self, img_size=256, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=64, layers=[4, 4, 4, 4], drop_rate=0.5,
                 norm_layer=nn.BatchNorm2d, alpha=3, use_dropout=False, patch_norm=True, **kwargs):
        super(MLPNet, self).__init__()

        self.num_classes = num_classes
        self.num_layers = len(layers)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patch_norm = patch_norm
        self.drop_rate = drop_rate
        self.patch1 = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=embed_dim,
            norm_layer=None)

        self.patch_unembed1 = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=embed_dim,
            norm_layer=None)
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size, bias=False)

        patches_resolution = [img_size // 2, img_size // 2]

        self.patches_resolution = patches_resolution

        self.avgpool = nn.MaxPool2d(2, 2)

        self.blocks0 = nn.ModuleList()
        for i in range(layers[0]):
            basic = BasicBlock(32, self.patches_resolution[0], self.patches_resolution[1], alpha,
                               use_dropout=use_dropout, drop_rate=drop_rate)
            self.blocks0.append(basic)



        self.blocks1 = nn.ModuleList()
        for i in range(layers[0]):
            basic = BasicBlock(embed_dim, int(self.patches_resolution[0] / 2 ), int(self.patches_resolution[1] / 2 ), alpha,
                               use_dropout=use_dropout, drop_rate=drop_rate)
            self.blocks1.append(basic)

        self.blocks2 = nn.ModuleList()
        for i in range(layers[1]):
            basic = BasicBlock(embed_dim * 2, int(self.patches_resolution[0] / 4), int(self.patches_resolution[1] / 4),
                               alpha, use_dropout=use_dropout, drop_rate=drop_rate)
            self.blocks2.append(basic)

        self.blocks3 = nn.ModuleList()
        for i in range(layers[2]):
            basic = BasicBlock(embed_dim * 4, int(self.patches_resolution[0] / 8), int(self.patches_resolution[1] / 8),
                               alpha, use_dropout=use_dropout, drop_rate=drop_rate)
            self.blocks3.append(basic)

        self.blocks4 = nn.ModuleList()
        for i in range(layers[3]):
            basic = BasicBlock(embed_dim * 8, int(self.patches_resolution[0] / 8), int(self.patches_resolution[1] / 8),
                               alpha, use_dropout=use_dropout, drop_rate=drop_rate)
            self.blocks4.append(basic)

        #self.merging1 = nn.Conv2d(embed_dim, embed_dim * 2, 2, 2, bias=False)
        self.stem = Res_CBAM_block(3, 16)
        self.RCB1 = nn.Sequential(nn.MaxPool2d(2, 2), Res_CBAM_block(16, 32))
        self.RCB2 = nn.Sequential(nn.MaxPool2d(2, 2), Res_CBAM_block(32, 64))
        self.RCB3 = nn.Sequential(nn.MaxPool2d(2, 2), Res_CBAM_block(64, 128))
        self.RCB4 = nn.Sequential(nn.MaxPool2d(2, 2), Res_CBAM_block(128, 256))







        self.patchmerging1 = PatchMerging(16)
        self.patchmerging2 = PatchMerging(32)
        self.patchmerging3 = PatchMerging(80)
        self.patchmerging4 = PatchMerging(160)












        self.m1 = nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=1, stride=1, bias=False)






        self.m2 = nn.Conv2d(embed_dim * 2, embed_dim * 4, kernel_size=1, stride=1, bias=False)
        self.conv_in2 = Res_CBAM_block(embed_dim * 4, embed_dim * 8)
        self.conv_in22 = Res_CBAM_block(embed_dim * 4, embed_dim * 4)
        self.coatt2 = CoordAtt(embed_dim * 4, embed_dim * 4)

        self.m3 = nn.Conv2d(embed_dim * 4, embed_dim * 8, kernel_size=1, stride=1, bias=False)
        self.conv_in3 = Res_CBAM_block(embed_dim * 4, embed_dim * 8)
        self.conv_in33 = Res_CBAM_block(embed_dim * 8, embed_dim * 8)
        self.merging4 = nn.ConvTranspose2d(embed_dim * 8, embed_dim * 4, 2, 2, bias=False)
        self.merging5 = nn.ConvTranspose2d(embed_dim * 4, embed_dim * 2, 2, 2, bias=False)
        self.merging6 = nn.ConvTranspose2d(embed_dim * 2, embed_dim, 2, 2, bias=False)

        self.up1 = nn.ConvTranspose2d(embed_dim * 8, embed_dim * 4, 2, 2, bias=False)
        self.up2 = nn.ConvTranspose2d(embed_dim * 4, embed_dim * 2, 2, 2, bias=False)
        self.up3 = nn.ConvTranspose2d(embed_dim * 2, embed_dim, 2, 2, bias=False)

        self.up4 = nn.ConvTranspose2d(embed_dim, 32, 2, 2, bias=False)
        self.up5 = nn.ConvTranspose2d(32, 32, 2, 2, bias=False)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fcn1 = _FCNHead(16, 1)
        self.fcn2 = _FCNHead(64, 1)
        self.fcn3 = _FCNHead(128, 1)
        self.fcn4 = _FCNHead(128, 1)
        self.fcn5 = _FCNHead(256, 1)

        self.UP = UpsampleBlock(32, 16)







        self.conv_s1_28 = nn.Conv2d(embed_dim * 2, embed_dim * 4, (2, 2), 2, 0, groups=embed_dim * 2, bias=False)  # depth-wise卷积160-320
        self.conv_s1_14 = nn.Conv2d(embed_dim * 4, embed_dim * 8, (2, 2), 2, 0, groups=embed_dim * 4, bias=False)#320-640
        self.conv_s2_14 = nn.Conv2d(embed_dim * 4, embed_dim * 8, (2, 2), 2, 0, groups=embed_dim * 4, bias=False)#320-640

        self.conv_s1_281 = nn.Conv2d(embed_dim * 4, embed_dim * 2, (2, 2), 2, 0, groups=embed_dim * 2, bias=False)  # depth-wise卷积320-160
        self.conv_s1_141 = nn.Conv2d(embed_dim * 8, embed_dim * 4, (2, 2), 2, 0, groups=embed_dim * 4, bias=False)#640-320
        self.conv_s2_141 = nn.Conv2d(embed_dim * 8, embed_dim * 4, (2, 2), 2, 0, groups=embed_dim * 4, bias=False)#640-320
        # self.conv_transpose = nn.ConvTranspose2d(in_channels=embed_dim,
        #                                          out_channels=1,
        #                                          kernel_size=patch_size,
        #                                          stride=patch_size,
        #                                          bias=False)
        self.conv1 = ConvLayer(16, 1, kernel_size=1, stride=1)
        self.convout = ConvLayer(32, 1, kernel_size=1, stride=1)
        self.CSFM0 = CSFM(16)
        self.CSFM1 = CSFM(32)
        self.CSFM2 = CSFM(embed_dim)
        self.CSFM3 = CSFM(embed_dim * 2)
        self.CSFM4 = CSFM(embed_dim * 4)
        self.head = nn.Linear(int(self.num_features), num_classes)



        self.norm = nn.BatchNorm2d(self.num_features)
        self.last_activation = nn.Sigmoid()

    def forward_features(self, x):

        x_0 = self.stem(x)              #16


        x1a = self.RCB1(x_0)#16--32
        x1b = self.PTIM_blocks(self.blocks0, x1a)
        x1 = x1a + x1b




        x2a = self.RCB2(x1) #32--64
        x2b = self.PTIM_blocks(self.blocks1, x2a)
        x2 = x2a + x2b


        x3a = self.RCB3(x2)    # 下采样2倍    128-256
        x3b = self.PTIM_blocks(self.blocks2, x3a)    #256
        x3 = x3a + x3b

        x4a = self.RCB4(x3)  #下采样 256-512
        x4b = self.PTIM_blocks(self.blocks3, x4a)  #512
        x4 = x4a + x4b




        return x1, x2, x3, x4, x_0

    def decoder(self, x1, x2, x3, x4, x_0):

        l_3 = self.CSFM3(x3, x4)  #256  32
        # l_3a = self.up3(l_3)




        l_2 = self.CSFM2(x2, l_3)  #128 64
        # l_2a = self.up2(l_2)
        # print(l_2a.shape)



        l_1 = self.CSFM1(x1, l_2)    #64 128

        l = self.UP(l_1) + x_0  #  8 16 256 256
        # print(l.shape)


        out = self.fcn1(l)



        return out

    def PTIM_blocks(self, blocks, x):
        for b in blocks:
            x = b(x)
        return x

    def forward(self, x):
        x1, x2, x3, x4, x_0 = self.forward_features(x)

        out = self.decoder(x1, x2, x3, x4, x_0)

        return [out.sigmoid()]



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# net = MLPNet().to(device)
# input_shape = (1, 3, 256, 256)  # 通常batch size是第一个维度
# input_tensor = torch.randn(input_shape).to(device)
#
# flops, params = profile(net, inputs=(input_tensor,))
# # print("FLOPs:", flops)
# # print("Params:", params)
#
# flops, params = clever_format([flops, params], "%.3f")
# print("FLOPs: %s" %(flops))
# print("params: %s" %(params))






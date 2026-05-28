import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
from torchsummary import summary
from  torchvision import models
from .conv import ResBlock
BatchNorm2d = nn.BatchNorm2d
from model.necks import FPN
BN_MOMENTUM = 0.01

class ResNet50FPN_Stride16(nn.Module):
    def __init__(self):
        super(ResNet50FPN_Stride16, self).__init__()
        resnet_50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.conv1 = resnet_50.conv1
        self.bn1 = resnet_50.bn1
        self.relu = resnet_50.relu
        self.maxpool = resnet_50.maxpool
        self.layer1 = resnet_50.layer1
        self.layer2 = resnet_50.layer2
        self.layer3 = resnet_50.layer3
        self.layer4 = resnet_50.layer4

        in_channels = [512,1024,2048]
        self.neck2f = FPN(in_channels, 256, len(in_channels))
        self.feature_head = nn.Sequential(
            nn.Dropout2d(0.2),
            ResBlock(in_dim=768, out_dim=384, dilation=0, norm="bn"),
            ResBlock(in_dim=384, out_dim=256, dilation=0, norm="bn"),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f_list = []
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        f_list.append(x2)
        x3 = self.layer3(x2)
        f_list.append(x3)
        x4 = self.layer4(x3)
        f_list.append(x4)

        fpn_f_list = self.neck2f(f_list)
        outputs = []
        outputs.append(F.interpolate(fpn_f_list[0],scale_factor=0.5, mode='bilinear', align_corners=True))
        outputs.append(fpn_f_list[1])
        outputs.append(F.interpolate(fpn_f_list[2],scale_factor=2, mode='bilinear', align_corners=True))
        multi_scale_f = torch.cat([outputs[0], outputs[1], outputs[2]], dim=1)
        feature = self.feature_head(multi_scale_f)
        outputs.append(feature)
        return outputs


class ResNet50FPN_Stride8(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50FPN_Stride8, self).__init__()

        # 1. 加载 ImageNet 预训练的 ResNet50 Backbone
        print(f"Loading ResNet50 ImageNet weights: {pretrained}")
        # 使用新版 torchvision 写法 (如果报错可改为 models.resnet50(pretrained=True))
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        # 提取 ResNet 的各层
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu1 = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1 # C2: Stride 4,  256 channels
        self.layer2 = resnet.layer2 # C3: Stride 8,  512 channels
        self.layer3 = resnet.layer3 # C4: Stride 16, 1024 channels
        self.layer4 = resnet.layer4 # C5: Stride 32, 2048 channels

        # 2. FPN 模块构建 (Conv + BN + ReLU)
        fpn_dim = 256

        # Top layer (处理 C5)
        self.toplayer = nn.Sequential(
            nn.Conv2d(2048, fpn_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )

        # Lateral layers (处理 C4, C3, C2)
        self.latlayer1 = nn.Sequential(
            nn.Conv2d(1024, fpn_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )
        self.latlayer2 = nn.Sequential(
            nn.Conv2d(512, fpn_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )
        self.latlayer3 = nn.Sequential(
            nn.Conv2d(256, fpn_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )

        # Smooth layers (消除混叠)
        self.smooth1 = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )
        self.smooth2 = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )
        self.smooth3 = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )

        # 3. 最终融合层 (Fusion to 256 channels)
        # 输入: 4个层级拼接 (256*4 = 1024) -> 输出: 256
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fpn_dim * 4, fpn_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, 256, kernel_size=1, stride=1, padding=0)
        )

        # 权重初始化 (Backbone 已经有预训练权重，只需初始化新加的层)
        self._init_new_layers()

    def _init_new_layers(self):
        # 仅初始化非 backbone 的部分
        for m in [self.toplayer, self.latlayer1, self.latlayer2, self.latlayer3,
                  self.smooth1, self.smooth2, self.smooth3, self.fusion_conv]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

    def _upsample_add(self, x, y):
        '''将 x 上采样并与 y 相加'''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # --- Bottom-up Backbone ---
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)   # Stride 4
        c3 = self.layer2(c2)  # Stride 8
        c4 = self.layer3(c3)  # Stride 16
        c5 = self.layer4(c4)  # Stride 32

        # --- Top-down FPN ---
        # P5
        p5 = self.toplayer(c5)
        
        # P4
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p4_smooth = self.smooth1(p4)

        # P3
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p3_smooth = self.smooth2(p3)

        # P2
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p2_smooth = self.smooth3(p2)

        # 此时:
        # p2_smooth: Stride 4
        # p3_smooth: Stride 8
        # p4_smooth: Stride 16
        # p5:        Stride 32 (注意: p5通常不加smooth conv，或者在输出前加，这里直接复用)

        # --- Multi-scale Fusion (Target: Stride 8) ---
        
        # 1. P2 (Stride 4) -> 下采样 -> Stride 8
        # 使用平均池化进行下采样
        feat_s8_p2 = F.avg_pool2d(p2_smooth, kernel_size=2, stride=2)

        # 2. P3 (Stride 8) -> 保持不变 -> Stride 8
        feat_s8_p3 = p3_smooth

        # 3. P4 (Stride 16) -> 上采样 2倍 -> Stride 8
        feat_s8_p4 = F.interpolate(p4_smooth, scale_factor=2, mode='bilinear', align_corners=False)

        # 4. P5 (Stride 32) -> 上采样 4倍 -> Stride 8
        feat_s8_p5 = F.interpolate(p5, scale_factor=4, mode='bilinear', align_corners=False)

        # --- Concatenation ---
        # [B, 256*4, H/8, W/8]
        out = torch.cat([feat_s8_p2, feat_s8_p3, feat_s8_p4, feat_s8_p5], dim=1)

        # --- Fusion Layer ---
        # [B, 1024, H/8, W/8] -> [B, 256, H/8, W/8]
        out = self.fusion_conv(out)

        return out
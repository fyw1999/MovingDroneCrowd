from misc.utils import *
from misc.layer import *
from .conv import ResBlock
from model.necks import FPN
from  torchvision import models
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01

class VGG16_FPN_Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG16_FPN_Encoder, self).__init__()

        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        features = list(vgg.features.children())

        self.layer1 = nn.Sequential(*features[0:23])
        self.layer2 = nn.Sequential(*features[23:33])
        self.layer3 = nn.Sequential(*features[33:43])

        in_channels = [256,512,512]
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
        f_list = []
        x1 = self.layer1(x)
        f_list.append(x1)
        x2 = self.layer2(x1)
        f_list.append(x2)
        x3 = self.layer3(x2)
        f_list.append(x3)

        fpn_f_list = self.neck2f(f_list)
        outputs = []
        outputs.append(F.interpolate(fpn_f_list[0], scale_factor=0.25, mode='bilinear', align_corners=True))
        outputs.append(F.interpolate(fpn_f_list[1], scale_factor=0.5, mode='bilinear', align_corners=True))
        outputs.append(fpn_f_list[2])
        multi_scale_f = torch.cat([outputs[0], outputs[1], outputs[2]], dim=1)
        feature = self.feature_head(multi_scale_f)
        outputs.append(feature)
        return outputs
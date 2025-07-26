from  torchvision import models
import sys
import torch.nn.functional as F
from misc.utils import *
from misc.layer import *
from torchsummary import summary
from model.necks import FPN
from .VGG.conv import ResBlock

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01


class GlobalDecoder(nn.Module):
    def __init__(self):
        super(GlobalDecoder, self).__init__()
        self.decode_head0 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, stride=1)
        )  

    def forward(self, x):
        x = F.interpolate(
                        self.decode_head0(x), scale_factor=2, mode='bilinear', align_corners=False)
        x = F.interpolate(
                        self.decode_head1(x), scale_factor=2, mode='bilinear', align_corners=False)
        x = F.interpolate(
                        self.decode_head2(x), scale_factor=2, mode='bilinear', align_corners=False)
        x = F.interpolate(
                        self.decode_head3(x), scale_factor=2, mode='bilinear', align_corners=False)
        return x


# class GlobalDecoder(nn.Module):
#     def __init__(self):
#         super(GlobalDecoder, self).__init__()
#         self.decode_head0 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.GroupNorm(8, 256),
#             nn.ReLU(inplace=True)
#         )
#         self.decode_head1 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.GroupNorm(8, 256),
#             nn.ReLU(inplace=True)
#         )
#         self.decode_head2 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.GroupNorm(8, 256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 1, kernel_size=1, stride=1)
#         )  

#     def forward(self, x):
#         x = F.interpolate(
#                         self.decode_head0(x), scale_factor=2, mode='bilinear', align_corners=False)
#         x = F.interpolate(
#                         self.decode_head1(x), scale_factor=2, mode='bilinear', align_corners=False)
#         x = F.interpolate(
#                         self.decode_head2(x), scale_factor=2, mode='bilinear', align_corners=False)
#         return x

# class GlobalDecoder(nn.Module):
#     def __init__(self):
#         super(GlobalDecoder, self).__init__()
#         self.decode_head0 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.GroupNorm(8, 256),
#             nn.ReLU(inplace=True)
#         )
#         self.decode_head1 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.GroupNorm(8, 256),
#             nn.ReLU(inplace=True)
#         )
#         self.decode_head2 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.GroupNorm(8, 256),
#             nn.ReLU(inplace=True)
#         )
#         self.decode_head3 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.GroupNorm(8, 256),
#             nn.ReLU(inplace=True)
#         )
#         self.decode_head4 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.GroupNorm(8, 256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 1, kernel_size=1, stride=1)
#         )  

#     def forward(self, x):
#         x = F.interpolate(
#                         self.decode_head0(x), scale_factor=2, mode='bilinear', align_corners=False)
#         x = F.interpolate(
#                         self.decode_head1(x), scale_factor=2, mode='bilinear', align_corners=False)
#         x = F.interpolate(
#                         self.decode_head2(x), scale_factor=2, mode='bilinear', align_corners=False)
#         x = F.interpolate(
#                         self.decode_head3(x), scale_factor=2, mode='bilinear', align_corners=False)
#         x = F.interpolate(
#                         self.decode_head4(x), scale_factor=2, mode='bilinear', align_corners=False)
#         return x
class ShareDecoder(nn.Module):
    def __init__(self):
        super(ShareDecoder, self).__init__()
        self.decode_head0 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, stride=1)
        )  

    def forward(self, x):
        x = F.interpolate(
                        self.decode_head0(x), scale_factor=2, mode='bilinear', align_corners=False)
        x = F.interpolate(
                        self.decode_head1(x), scale_factor=2, mode='bilinear', align_corners=False)
        x = F.interpolate(
                        self.decode_head2(x), scale_factor=2, mode='bilinear', align_corners=False)
        x = F.interpolate(
                        self.decode_head3(x), scale_factor=2, mode='bilinear', align_corners=False)
        return x

# class ShareDecoder(nn.Module):
#     def __init__(self):
#         super(ShareDecoder, self).__init__()
#         self.decode_head0 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.GroupNorm(8, 256),
#             nn.ReLU(inplace=True)
#         )
#         self.decode_head1 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.GroupNorm(8, 256),
#             nn.ReLU(inplace=True)
#         )
#         self.decode_head2 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.GroupNorm(8, 256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 1, kernel_size=1, stride=1)
#         )  

#     def forward(self, x):
#         x = F.interpolate(
#                         self.decode_head0(x), scale_factor=2, mode='bilinear', align_corners=False)
#         x = F.interpolate(
#                         self.decode_head1(x), scale_factor=2, mode='bilinear', align_corners=False)
#         x = F.interpolate(
#                         self.decode_head2(x), scale_factor=2, mode='bilinear', align_corners=False)
#         return x
    

# class ShareDecoder(nn.Module):
#     def __init__(self):
#         super(ShareDecoder, self).__init__()
#         self.decode_head0 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.GroupNorm(8, 256),
#             nn.ReLU(inplace=True)
#         )
#         self.decode_head1 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.GroupNorm(8, 256),
#             nn.ReLU(inplace=True)
#         )
#         self.decode_head2 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.GroupNorm(8, 256),
#             nn.ReLU(inplace=True)
#         )
#         self.decode_head3 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.GroupNorm(8, 256),
#             nn.ReLU(inplace=True)
#         )
#         self.decode_head4 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.GroupNorm(8, 256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 1, kernel_size=1, stride=1)
#         )  

#     def forward(self, x):
#         x = F.interpolate(
#                         self.decode_head0(x), scale_factor=2, mode='bilinear', align_corners=False)
#         x = F.interpolate(
#                         self.decode_head1(x), scale_factor=2, mode='bilinear', align_corners=False)
#         x = F.interpolate(
#                         self.decode_head2(x), scale_factor=2, mode='bilinear', align_corners=False)
#         x = F.interpolate(
#                         self.decode_head3(x), scale_factor=2, mode='bilinear', align_corners=False)
#         x = F.interpolate(
#                         self.decode_head4(x), scale_factor=2, mode='bilinear', align_corners=False)
#         return x
# class InOutDecoder(nn.Module):
#     def __init__(self):
#         super(InOutDecoder, self).__init__()
#         self.loc_head = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(16, momentum=BN_MOMENTUM),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(inplace=True)
#         )
        
#     def forward(self, x):
#         x = self.loc_head(x)
#         return x

class InOutDecoder(nn.Module):
    def __init__(self):
        super(InOutDecoder, self).__init__()
        self.loc_head = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, stride=1)
        )
        
    def forward(self, x):
        x = self.loc_head(x)
        return x




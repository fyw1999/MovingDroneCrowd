import torch.nn.functional as F
from torch import nn

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

class GlobalDecoder_Stride8(nn.Module):
    def __init__(self):
        super(GlobalDecoder_Stride8, self).__init__()
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
        return x

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




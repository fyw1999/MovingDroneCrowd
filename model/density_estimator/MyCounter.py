import torch.nn as nn
from model.VGG.VGG16_FPN import VGG16FPN_Stride8
from model.decoder import GlobalDecoder_Stride8
class CustomedCounter(nn.Module):
    def __init__(self, froze=True):
        super(CustomedCounter, self).__init__()
        self.Extractor = VGG16FPN_Stride8()
        self.global_decoder = GlobalDecoder_Stride8()
        if froze: self.frozePara()

    def frozePara(self):
        for para in self.parameters():
            para.requires_grad = False

    def forward(self,x):
        features = self.Extractor(x)
        if self.training:
            density_maps = self.global_decoder(features)
        else:
            density_maps = self.global_decoder(features)/200
        return density_maps
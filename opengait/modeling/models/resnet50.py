import torch

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks

from torchvision.models.resnet import BasicBlock, ResNet
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange


class ResNet50Backbone(ResNet):
    def __init__(self):
        super(ResNet50Backbone, self).__init__(BasicBlock, [3, 4, 6, 3])
        layers = [3, 4, 6, 3]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], dilate=False)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], dilate=False)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], dilate=False)
    def forward(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # x = torch.Size([1920, 1, 64, 64]) (n*s, c, h, w)
        x = self.conv1(x) # torch.Size([1920, 64, 32, 32]) (n*s, c=64, h, w)
        x = self.bn1(x) # torch.Size([1920, 64, 32, 32])
        x = self.relu(x) # torch.Size([1920, 64, 32, 32])
        x = self.maxpool(x) # torch.Size([1920, 64, 16, 16])

        x = self.layer1(x) # torch.Size([1920, 64, 16, 16])
        x = self.layer2(x) # torch.Size([1920, 128, 16, 16])
        x = self.layer3(x) # torch.Size([1920, 256, 16, 16])
        x = self.layer4(x) # torch.Size([1920, 512, 16, 16]) (n*s, c=512, h, w)
        x = x.reshape(-1, 30, 512, 16, 16).transpose(1, 2).contiguous() # (n, c, s, h, w) = torch.Size([64, 512, 30, 16, 16]),

        x = torch.max(x, dim=2) # (n, c, h, w) = torch.Size([64, 512, 16, 16]) butun pikseller için framelerdeki max nokta
    
        return x

class ResNet50(BaseModel):

    def build_network(self, model_cfg):
        self.Backbone = ResNet50Backbone()
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])
        self.FCs = SeparateFCs(parts_num=16, in_channels=512, out_channels=256)
        self.BNNecks = SeparateBNNecks(parts_num=16, class_num=74, in_channels=256)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        

        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        else:
            sils = rearrange(sils, 'n s c h w -> n c s h w')

        del ipts
        x = sils.reshape(-1, 1, 64, 64)
        outs = self.Backbone(x)  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
        embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils,'n c s h w -> (n s) c h w')
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval
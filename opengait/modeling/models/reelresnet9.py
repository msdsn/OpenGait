import torch

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks

from einops import rearrange
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn as nn

class ReelResNet9(BaseModel):
    def build_network(self, model_cfg):
        self.resnet9 = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=75)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(2) # n s c h w
        n, s, c, h, w = sils.size()
        x = rearrange(sils, 'n s c h w -> (n s) c h w')
        # x = n*s, c=1, h=64, w=64
        x = self.conv1(x) # n*s, c=64, h=32, w= 32
        x = self.resnet9.bn1(x) # n*s, c=64, h=32, w= 32
        x = self.resnet9.relu(x) # n*s, c=64, h=32, w= 32
        x = self.resnet9.maxpool(x) # n*s, c=64, h=16, w= 16
        x = self.resnet9.layer1(x) # n*s, c=128, h=8, w= 8
        x = self.resnet9.layer2(x) # n*s, c=256, h=4, w= 4
        x = self.resnet9.layer3(x) # n*s, c=512, h=2, w= 2
        x = self.resnet9.layer4(x) # n*s, c=512, h=1, w= 1

        # Global Average Pooling
        x = self.resnet9.avgpool(x) # n*s, c=512 
        flatten = torch.flatten(x, 1)
        logits = self.resnet9.fc(flatten) # n*s, c=75 
        logits = rearrange(logits, '(n s) c -> n c s', n=n, s=s)
        embed = rearrange(flatten, '(n s) c -> n c s', n=n, s=s)
            
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
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
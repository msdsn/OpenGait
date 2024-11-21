import torch

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks

from einops import rearrange

from torchvision.models.vision_transformer import Encoder
import torch.nn as nn
import torch.nn.functional as F

class BaseAttention(BaseModel):

    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        
        self.conv_reduce = nn.Conv2d(512, 128, kernel_size=2, stride=2, padding=0, bias=False)
        self.encoder = Encoder(**model_cfg['Encoder'])
        self.class_token = nn.Parameter(torch.zeros(1, 1, 128))
        self.conv_expand = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False)

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

        self.fusion_conv = nn.Conv2d(512+512, 512, kernel_size=1, stride=1, padding=0, bias=False)

        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        else:
            sils = rearrange(sils, 'n s c h w -> n c s h w')

        del ipts
        outs = self.Backbone(sils)  # [n, c, s, h, w]

        # ---- Path (Encoder) ----
        n, c, s, h, w = outs.size()
        x1 = rearrange(outs, 'n c s h w -> (n s) c h w')
        x1 = self.conv_reduce(x1)
        x1 = rearrange(x1, '(n s) c h w -> (n h w) s c', n=n, s=s)
        batch_class_token = self.class_token.expand(n*h*w, -1, -1)
        x1 = torch.cat((batch_class_token, x1), dim=1)
        x1 = self.encoder(x1)
        x1 = x1[:, 0, :] # [n*h*w, c]
        x1 = rearrange(x1, '(n h w) c -> n c h w', n=n, h=h, w=w)
        x1 = self.conv_expand(x1) # [n, c, h, w]
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)

        # ---- Path (Pooling) ----
        x2 = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]

        #[Pooling]       [Encoder]
        #   |                |
        #   +-->  Fusion  <--+
        #           |
        #           V
        # [Horizontal Pooling Matching]

        # Fusion
        outs = torch.cat((x1, x2), dim=1)
        outs = self.fusion_conv(outs)

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
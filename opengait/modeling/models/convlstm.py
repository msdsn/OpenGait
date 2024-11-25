import torch

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks

from einops import rearrange

from torchvision.models.vision_transformer import Encoder
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTM(BaseModel):

    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])
        
        self.fusion_conv = nn.Conv2d(512+256, 256, kernel_size=1, stride=1, padding=0, bias=False)
        
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

        # [n*h*w, s, c]
        # ---- Path (Encoder) ----
        n, c, s, h, w = outs.size()
        x1 = rearrange(outs, 'n c s h w -> (n h w) s c')
        x1, _ = self.lstm(x1)  # x: [n*h*w, s, hidden_size]
        x1 = x1[:, -1, :]  # [n*h*w, hidden_size]
        x1 = rearrange(x1, '(n h w) c -> n c h w', n=n, h=h, w=w)

        # ---- Path (Pooling) ----
        x2 = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]

        #[Pooling]       [LSTM]
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
        embed_2, logits = self.BNNecks(feat)  # [n, c, p]
        embed = feat

        retval = {
            'training_feat': {
                'triplet': {'embeddings': feat, 'labels': labs},
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
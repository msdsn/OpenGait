import torch

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks

from einops import rearrange

from torchvision.models.vision_transformer import Encoder
import torch.nn as nn
import torch.nn.functional as F

class BaseLSTM(BaseModel):

    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        
        self.lstm = nn.LSTM(input_size=512, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        for param in self.lstm.parameters():
            param.requires_grad = False
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])
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

        if self.iteration < 24000:
            # ---- Path (Pooling) ----
            x = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        else:
            for param in self.lstm.parameters():
                param.requires_grad = True
            n, c, s, h, w = outs.size()
            x = rearrange(outs, 'n c s h w -> (n h w) s c')
            x, _ = self.lstm(x)  # x: [n*h*w, s, hidden_size]
            x = x[:, -1, :]  # [n*h*w, hidden_size]
            x = rearrange(x, '(n h w) c -> n c h w', n=n, h=h, w=w)
            

        # Horizontal Pooling Matching, HPM
        feat = self.HPP(x)  # [n, c, p]
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
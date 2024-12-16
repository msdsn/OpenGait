import torch

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid2, PackSequenceWrapper, SeparateFCs, SeparateBNNecks

from einops import rearrange

import torch.nn as nn
import torch.nn.functional as F

class MHAPartLSTM(BaseModel):

    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid2(bin_num=model_cfg['bin_num'])
        
        self.fusion_conv = nn.Conv1d(512+256, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.fusion_bn = nn.BatchNorm1d(256)  # Batch Normalization katmanı eklendi
        self.mha = nn.MultiheadAttention(embed_dim=256, num_heads=8)  # MHA katmanı eklendi
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

        part = self.HPP(outs)  # [n, c, s, p]
        
        # ---- Path (LSTM) ----
        n, c, s, p = part.size()
        x1 = rearrange(part, 'n c s p -> (n p) s c')
        x1, _ = self.lstm(x1)  # x: [n*p, s, hidden_size]
        x1 = x1[:, -1, :]  # [n*p, hidden_size]
        x1 = rearrange(x1, '(n p) c -> n c p', n=n, p=p)

        # ---- Path (Pooling) ----
        x2 = self.TP(part, seqL, options={"dim": 2})[0]  # [n, c, p]

        # Fusion
        outs = torch.cat((x1, x2), dim=1) # [n, c+hidden_size, p]
        feat = self.fusion_conv(outs)
        feat = self.fusion_bn(feat) 
        # ---- Multi-Head Attention Uygulama ----
        # feat: [n, 256, p] -> [p, n, 256]
        feat = feat.permute(2, 0, 1).contiguous()  # feat: [p, n, 256]
        attn_output, attn_weights = self.mha(feat, feat, feat)  # [p, n, 256]
        feat = attn_output.permute(1, 2, 0).contiguous()  # [n, 256, p]

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
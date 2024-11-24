import torch

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks

from einops import rearrange


import torch.nn as nn
import torch.nn.functional as F

class AttentionLSTM(BaseModel):

    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])
        
        # Attention-based fusion
        self.attn_fusion = AttentionFusion(in_channels1=256, in_channels2=512, out_channels=512)

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

        
        # ---- Path (LSTM) ----
        n, c, s, h, w = outs.size()
        x1 = rearrange(outs, 'n c s h w -> (n h w) s c')
        x1, _ = self.lstm(x1)  # x: [n*h*w, s, hidden_size]
        x1 = x1[:, -1, :]  # [n*h*w, hidden_size]
        x1 = rearrange(x1, '(n h w) c -> n c h w', n=n, h=h, w=w)

        # ---- Path (Pooling) ----
        x2 = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]

        #[Pooling]            [LSTM]
        #   |                   |
        #   +-->  ATTFusion  <--+
        #           |
        #           V
        # [Horizontal Pooling Matching]

        # Fusion
        # Attention-based Fusion
        x_fused = self.attn_fusion(x1, x2)  # [n, 256, h, w]

        # Horizontal Pooling Matching, HPM
        feat = self.HPP(x_fused)  # [n, c, p]

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
    



# Define the AttentionFusion module
class AttentionFusion(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(AttentionFusion, self).__init__()
        self.in_channels1 = in_channels1
        self.in_channels2 = in_channels2
        total_channels = in_channels1 + in_channels2

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(total_channels, total_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels // 2, total_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        # Output convolution
        self.output_conv = nn.Conv2d(total_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2):
        # Concatenate features from both paths
        x_concat = torch.cat([x1, x2], dim=1)  # [n, c1 + c2, h, w]

        # Compute attention weights
        attn_weights = self.attention(x_concat)  # [n, c1 + c2, h, w]

        # Apply attention weights
        x_attended = x_concat * attn_weights

        # Fuse features
        x_fused = self.output_conv(x_attended)
        return x_fused
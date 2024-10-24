import torch

from ..base_model import BaseModel
from ..modules import  SeparateFCs, SeparateBNNecks

import torch
from torchvision.models import VisionTransformer
import torch.nn as nn
from einops import rearrange

class Vit(BaseModel):

    def build_network(self, model_cfg):
        self.Backbone = VisionTransformerB16()

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        else:
            sils = rearrange(sils, 'n s c h w -> n c s h w')

            

        del ipts
        x = sils.reshape(-1, 1, 64, 64)
        outs = self.Backbone(x)  # [n, c, s, h, w]


        retval = {
            'training_feat': {
                'triplet': {'embeddings': outs['triplet'], 'labels': labs},
                'softmax': {'logits': outs['softmax'], 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils,'n c s h w -> (n s) c h w')
            },
            'inference_feat': {
                'embeddings': outs['triplet']
            }
        }
        return retval
    





class VisionTransformerB16(VisionTransformer):
    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 16,
        num_layers: int = 12,
        num_heads: int = 12,
        hidden_dim: int = 768,
        mlp_dim: int = 3072,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 74,
    ):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_classes=num_classes,
        )
        # Giriş kanal sayısını 1'e düşür
        self.conv_proj = nn.Conv2d(
            in_channels=1,  # Tek kanal
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.FCs = SeparateFCs(parts_num=16, in_channels=hidden_dim, out_channels=256)
        self.BNNecks = SeparateBNNecks(parts_num=16, class_num=num_classes, in_channels=256)
    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        print(f"incoming x.shape: {x.shape}")
        # Reshape and permute the input tensor
        # [1920, 1, 64, 64] -> [1920, 16, 768] (h/patch_size)=4  (dikey parca*yatay parca)=16 hidden_dim=768
        x = self._process_input(x)
        print(f"x.shape after _process_input: {x.shape}")
        n = x.shape[0] # 1920

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1) # [1920, 1, 768] her sample icin class token
        x = torch.cat([batch_class_token, x], dim=1) # [1920, 16, 768] -> [1920, 17, 768]
        print(f"x.shape after cat: {x.shape}")
        x = self.encoder(x) # [1920, 17, 768]
        print(f"x.shape after encoder: {x.shape}")
        x = x[:, 1:] # [1920, 16, 768]
        print(f"x.shape after slicing: {x.shape}")
        # [n, p, c] -> [n, c, p]
        x = x.permute(0, 2, 1).contiguous() # [n, c=768, p]
        print(f"x.shape after permute: {x.shape}")
        x = x.reshape(-1, 30, x.size()[1], 16).transpose(1, 2).contiguous() # (n, c, s, p) = torch.Size([64, 768, 30, 16]),
        x = torch.max(x, dim=2)[0] # (n, c, p) = torch.Size([64, 768, 16]) butun pikseller için framelerdeki max nokta
        embed = self.FCs(x)  # [n, c=256, p]
        embed_2, logits = self.BNNecks(embed)  # [n, c, p]
        return {
                'triplet': embed,
                'softmax': logits
        }


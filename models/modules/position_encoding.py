# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        mask = torch.zeros((x.shape[0], x.shape[-1]), dtype=torch.bool, device=x.device)# B x W (x size is  # [b 1024(d) 256(w)])
        not_mask = ~mask
        x_embed = not_mask.cumsum(1, dtype=torch.float32) # B x W
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats) # N

        pos_x = x_embed[:, :,  None] / dim_t # B x W x N
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2) # B x W x N
        pos = pos_x.permute(0, 2, 1) # B x N x W
        return pos


def build_position_encoding_1d(hidden_dim, position_embedding_type):
    N_steps = hidden_dim
    if position_embedding_type in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    else:
        raise ValueError(f"not supported {position_embedding_type}")

    return position_embedding


if __name__ == '__main__':
    token_dim = 1024
    toke_len = 256

    transformer = PositionEmbeddingSine(toke_len, normalize=True)

    input = torch.randn(2, toke_len, token_dim)
    # query = nn.Embedding(toke_len, token_dim)
    output = transformer(input)
    print(output.shape)
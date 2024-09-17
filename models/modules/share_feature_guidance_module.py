import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from models.modules.transformer_modules import *


class Share_Feature_Guidance_Module(nn.Module):
    def __init__(self, dim, depth, heads, win_size, dim_head, mlp_dim,
                 dropout=0., patch_num=None, ape=None, rpe=None, rpe_pos=1):
        super().__init__()
        self.absolute_pos_embed = None if patch_num is None or ape is None else AbsolutePosition(dim, dropout,
                                                                                                 patch_num, ape)
        self.pos_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([])
        
        for i in range(depth):
            # cross attention
            cross_attention = CrossAttentionLayer(d_model=dim, nhead=heads, normalize_before=True)
            if i % 2 == 0:
                attention = WinAttention(dim, win_size=win_size, shift=0 if (i % 3 == 0) else win_size // 2,
                                         heads=heads, dim_head=dim_head, dropout=dropout, rpe=rpe, rpe_pos=rpe_pos)
            else:
                attention = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout,
                                      patch_num=patch_num, rpe=rpe, rpe_pos=rpe_pos)

            self.layers.append(nn.ModuleList([
                cross_attention,
                PreNorm(dim, attention),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ]))

    def forward(self, x, pos, query_embed, query_pos):
        if self.absolute_pos_embed is not None:
            x = self.absolute_pos_embed(x)
        x = self.pos_dropout(x)
        # handle embeddings
        bs, c, w = x.shape
        query_embeds = query_embed.unsqueeze(0).repeat(bs, 1, 1)
        query_pos_embeds = query_pos.unsqueeze(0).repeat(bs, 1, 1)
        # output = query_embeds
        output = x
        for cross, attn, ff in self.layers:
            # cross attention
            # Q = resnet feature
            # K, V = query
            # output = cross(tgt=output, memory=x, pos=pos, query_pos=query_pos_embeds)
            output = cross(tgt=output, memory=query_embeds, pos=query_pos_embeds, query_pos=pos)
            output = attn(output) + output
            output = ff(output) + output
        return output


if __name__ == '__main__':
    token_dim = 1024
    toke_len = 256

    transformer = Share_Feature_Guidance_Module(dim=token_dim,
                                  depth=6,
                                  heads=16,
                                  win_size=8,
                                  dim_head=64,
                                  mlp_dim=2048,
                                  dropout=0.1)

    input = torch.randn(2, toke_len, token_dim)
    query = nn.Embedding(512, token_dim)
    output = transformer(input, input, query.weight, query.weight)
    print(output.shape)

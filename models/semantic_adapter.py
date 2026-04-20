import torch
import torch.nn as nn


class SemanticAdapter(nn.Module):
    def __init__(self, clip_dim=512, embed_dim=384):
        super().__init__()
        self.affine = nn.Sequential(
            nn.LayerNorm(clip_dim),
            nn.Linear(clip_dim, embed_dim * 2),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(clip_dim),
            nn.Linear(clip_dim, embed_dim),
            nn.Sigmoid(),
        )

        nn.init.zeros_(self.affine[1].weight)
        nn.init.zeros_(self.affine[1].bias)
        nn.init.zeros_(self.gate[1].weight)
        nn.init.zeros_(self.gate[1].bias)

    def forward(self, decoder_feats, clip_feat):
        feat = decoder_feats[0]
        gamma, beta = self.affine(clip_feat).chunk(2, dim=-1)
        gate = self.gate(clip_feat)

        gamma = 0.15 * torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = 0.15 * beta.unsqueeze(-1).unsqueeze(-1)
        gate = gate.unsqueeze(-1).unsqueeze(-1)

        modulated = feat * (1.0 + gamma) + beta
        enhanced_feat = feat + gate * (modulated - feat)
        return [enhanced_feat]

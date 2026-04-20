import math

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.zoo import bmshj2018_factorized

from configs.base_config import (
    CLIP_EMBED_DIM,
    DEVICE,
    MODEL_ARCH_VERSION,
    MSE_LOSS_WEIGHT,
    PERCEPTUAL_LOSS_WEIGHT,
    VIT_PRETRAINED,
    VIT_QUALITY,
)
from .semantic_adapter import SemanticAdapter


class SemanticResidualBlock(nn.Module):
    def __init__(self, channels, clip_dim):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.film = nn.Sequential(
            nn.LayerNorm(clip_dim),
            nn.Linear(clip_dim, channels * 2),
        )
        nn.init.zeros_(self.film[1].weight)
        nn.init.zeros_(self.film[1].bias)

    def forward(self, x, clip_feat):
        gamma, beta = self.film(clip_feat).chunk(2, dim=-1)
        gamma = 0.25 * torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = 0.25 * beta.unsqueeze(-1).unsqueeze(-1)

        residual = self.norm(x)
        residual = residual * (1.0 + gamma) + beta
        residual = F.gelu(self.conv1(residual))
        residual = self.conv2(residual)
        return x + residual


class SemanticRefiner(nn.Module):
    def __init__(self, clip_dim):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.block1 = SemanticResidualBlock(96, clip_dim)
        self.block2 = SemanticResidualBlock(96, clip_dim)
        self.block3 = SemanticResidualBlock(96, clip_dim)
        self.block4 = SemanticResidualBlock(96, clip_dim)
        self.head = nn.Conv2d(96, 3, kernel_size=3, padding=1)

        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, clip_feat):
        feat = self.stem(x)
        feat = self.block1(feat, clip_feat)
        feat = self.block2(feat, clip_feat)
        feat = self.block3(feat, clip_feat)
        feat = self.block4(feat, clip_feat)
        residual = 0.30 * torch.tanh(self.head(feat))
        return (x + residual).clamp(0.0, 1.0)


class EnhancedViTCompressor(nn.Module):
    arch_version = MODEL_ARCH_VERSION

    def __init__(
        self,
        use_semantic=True,
        semantic_input_mode="clip",
        quality=VIT_QUALITY,
        pretrained=VIT_PRETRAINED,
        embed_dim=None,
    ):
        super().__init__()
        self.use_semantic = use_semantic
        self.semantic_input_mode = semantic_input_mode
        self.quality = quality
        self.pretrained = pretrained
        self.dtype = torch.float32
        self.downsample_factor = 16

        self.codec = bmshj2018_factorized(quality=quality, metric="mse", pretrained=pretrained, progress=False)
        self.embed_dim = self.codec.g_a[-1].out_channels if embed_dim is None else embed_dim

        if self.use_semantic:
            self.semantic_adapter = SemanticAdapter(clip_dim=CLIP_EMBED_DIM, embed_dim=self.embed_dim)
            self.semantic_refiner = SemanticRefiner(clip_dim=CLIP_EMBED_DIM)

        self.mse_loss = nn.MSELoss()
        self.lpips_loss_fn = lpips.LPIPS(net="vgg").to(DEVICE, dtype=self.dtype)
        self.lpips_loss_fn.eval()
        for param in self.lpips_loss_fn.parameters():
            param.requires_grad_(False)

        self.to(DEVICE, dtype=self.dtype)

    def _pad_to_multiple(self, x):
        _, _, height, width = x.shape
        padded_h = math.ceil(height / self.downsample_factor) * self.downsample_factor
        padded_w = math.ceil(width / self.downsample_factor) * self.downsample_factor
        pad_bottom = padded_h - height
        pad_right = padded_w - width
        if pad_bottom == 0 and pad_right == 0:
            return x, (height, width)
        x = F.pad(x, (0, pad_right, 0, pad_bottom), mode="replicate")
        return x, (height, width)

    def _crop_to_size(self, x, output_size):
        target_h, target_w = output_size
        return x[..., :target_h, :target_w]

    def encode(self, x):
        x = x.to(dtype=self.dtype)
        return self.codec.g_a(x)

    def decode(self, z_hat):
        return self.codec.g_s(z_hat.to(dtype=self.dtype))

    def decode_with_semantic(self, z_hat, clip_feat):
        enhanced_feats = self.semantic_adapter([z_hat.to(dtype=self.dtype)], clip_feat.to(dtype=self.dtype))
        x_hat = self.codec.g_s(enhanced_feats[0])
        return self.semantic_refiner(x_hat, clip_feat.to(dtype=self.dtype))

    def aux_loss(self):
        return self.codec.aux_loss()

    def main_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad and not name.endswith(".quantiles"):
                yield param

    def aux_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad and name.endswith(".quantiles"):
                yield param

    def loss(self, original_img, recon_img, likelihoods):
        original_img = original_img.to(DEVICE, dtype=self.dtype)
        recon_img = recon_img.to(DEVICE, dtype=self.dtype)

        original_01 = ((original_img + 1.0) / 2.0).clamp(0.0, 1.0)
        recon_01 = ((recon_img + 1.0) / 2.0).clamp(0.0, 1.0)

        perceptual_loss = self.lpips_loss_fn(original_01 * 2.0 - 1.0, recon_01 * 2.0 - 1.0).mean()
        mse_loss = self.mse_loss(original_01, recon_01)
        recon_loss = PERCEPTUAL_LOSS_WEIGHT * perceptual_loss + MSE_LOSS_WEIGHT * mse_loss

        num_pixels = original_img.shape[0] * original_img.shape[2] * original_img.shape[3]
        bpp_loss = sum(torch.log(likelihood).sum() / (-math.log(2) * num_pixels) for likelihood in likelihoods.values())
        total_loss = recon_loss + bpp_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "perceptual_loss": perceptual_loss,
            "mse_loss": mse_loss,
            "bpp": bpp_loss,
        }

    def forward(self, x, clip_feature=None):
        x = x.to(DEVICE, dtype=self.dtype)
        x_01 = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
        x_padded, output_size = self._pad_to_multiple(x_01)

        y = self.encode(x_padded)
        y_hat, y_likelihoods = self.codec.entropy_bottleneck(y)

        if self.use_semantic:
            if clip_feature is None:
                if self.semantic_input_mode == "zeros":
                    clip_feature = torch.zeros(
                        x.shape[0],
                        CLIP_EMBED_DIM,
                        device=DEVICE,
                        dtype=self.dtype,
                    )
                else:
                    raise ValueError("clip_feature is required when use_semantic=True")
            x_hat = self.decode_with_semantic(y_hat, clip_feature)
        else:
            x_hat = self.decode(y_hat)

        x_hat = self._crop_to_size(x_hat, output_size).clamp(0.0, 1.0)
        recon_img = x_hat * 2.0 - 1.0

        loss_dict = self.loss(x, recon_img, {"y": y_likelihoods})
        return recon_img, loss_dict

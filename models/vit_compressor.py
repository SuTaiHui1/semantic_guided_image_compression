import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips  # 确保提前安装：pip install lpips
from configs.base_config import DEVICE, VIT_QUALITY, VIT_PRETRAINED

# 定义ViT基础模块（Encoder/Decoder）
class ViTEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=384, patch_size=16, num_heads=6, num_layers=6):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch Embedding：将图像分块并映射到嵌入维度
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size, padding=0
        )
        
        # Transformer Encoder层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 归一化层
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, 3, H, W]
        B, C, H, W = x.shape
        
        # Patch Embedding：[B, 384, H//16, W//16]
        x = self.patch_embed(x)  # [B, embed_dim, H/patch, W/patch]
        H_patch, W_patch = x.shape[2], x.shape[3]
        
        # 展平为序列：[B, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        x = self.norm(x)
        
        # 恢复空间维度：[B, 384, H//16, W//16]
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H_patch, W_patch)
        return x

class ViTDecoder(nn.Module):
    def __init__(self, embed_dim=384, out_channels=3, patch_size=16, num_heads=6, num_layers=6):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Transformer Decoder层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 归一化层
        self.norm = nn.LayerNorm(embed_dim)
        
        # 反卷积：将嵌入维度映射回3通道图像
        self.head = nn.ConvTranspose2d(
            embed_dim, out_channels, 
            kernel_size=patch_size, stride=patch_size, padding=0
        )
        
        # 输出激活（将值限制在[-1,1]，匹配输入图像范围）
        self.act = nn.Tanh()

    def forward(self, x):
        # x: [B, 384, H//16, W//16]
        B, C, H_patch, W_patch = x.shape
        
        # 展平为序列：[B, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        
        # Transformer解码（目标序列和输入序列相同）
        x = self.transformer_decoder(x, x)
        x = self.norm(x)
        
        # 恢复空间维度：[B, 384, H//16, W//16]
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H_patch, W_patch)
        
        # 反卷积重建图像：[B, 3, H, W]
        x = self.head(x)
        x = self.act(x)  # 输出范围[-1,1]
        return x

# 核心ViT压缩模型类
class ViTCompressor(nn.Module):
    def __init__(self, quality=VIT_QUALITY, pretrained=VIT_PRETRAINED):
        super().__init__()
        self.quality = quality
        self.pretrained = pretrained
        self.dtype = torch.float32
        
        # 根据quality调整嵌入维度（可选，这里固定为384）
        embed_dim = 384
        patch_size = 16
        
        # 编码器和解码器
        self.encoder = ViTEncoder(
            in_channels=3, embed_dim=embed_dim, patch_size=patch_size
        )
        self.decoder = ViTDecoder(
            embed_dim=embed_dim, out_channels=3, patch_size=patch_size
        )
        
        # 初始化LPIPS（只创建一次，避免重复初始化）
        self.lpips_loss_fn = lpips.LPIPS(net='vgg').to(DEVICE, dtype=self.dtype)
        self.lpips_loss_fn.eval()  # LPIPS固定权重，不参与训练
        
        # 移动到指定设备
        self.to(DEVICE, dtype=self.dtype)

    def encode(self, x):
        """编码：原始图像 → 压缩特征"""
        x = x.to(dtype=self.dtype)
        z_hat = self.encoder(x)
        return z_hat

    def decode(self, z_hat):
        """解码：压缩特征 → 重建图像（Baseline模式）"""
        z_hat = z_hat.to(dtype=self.dtype)
        y_hat = self.decoder(z_hat)
        return y_hat

    def decode_with_semantic(self, z_hat, clip_feat, semantic_adapter):
        """
        解码：带语义融合的重建（增强模型模式）
        修正：参数顺序匹配SemanticAdapter的forward方法
        """
        z_hat = z_hat.to(dtype=self.dtype)
        clip_feat = clip_feat.to(dtype=self.dtype)
        
        # 提取解码器中间特征（适配SemanticAdapter的维度）
        decoder_feats = [z_hat]  # 单特征层，维度384
        # 修正调用参数顺序：decoder_feats在前，clip_feat在后
        enhanced_feats = semantic_adapter(decoder_feats, clip_feat)
        
        # 用融合后的特征解码
        y_hat = self.decoder(enhanced_feats[0])
        return y_hat

    def loss(self, original_img, recon_img, z_hat):
        """
        计算率失真损失（仅训练阶段使用）
        参数说明：
        - original_img: 原始3通道图像 [B,3,H,W]（输入图像，范围[-1,1]）
        - recon_img: 重建3通道图像 [B,3,H,W]（模型输出，范围[-1,1]）
        - z_hat: 编码器输出特征 [B,384,H//16,W//16]
        """
        # 统一数据类型和设备
        original_img = original_img.to(DEVICE, dtype=self.dtype)
        recon_img = recon_img.to(DEVICE, dtype=self.dtype)
        z_hat = z_hat.to(DEVICE, dtype=self.dtype)
        
        # 1. 重建损失：LPIPS（感知损失） + MSE（像素损失）
        # LPIPS要求输入范围[-1,1]（和我们的图像范围一致），直接计算
        with torch.no_grad():  # 禁用LPIPS梯度，避免更新其权重
            perceptual_loss = self.lpips_loss_fn(original_img, recon_img).mean()
        
        # MSE像素损失（衡量像素级误差）
        mse_loss = nn.MSELoss()(original_img, recon_img)
        
        # 平衡感知损失和像素损失（可根据效果调整权重）
        recon_loss = 0.1 * perceptual_loss + 0.9 * mse_loss
        
        # 2. 率损失（比特率损失，简化版）
        # 真实场景需用compressai的熵瓶颈计算BPP，这里简化为特征的L2范数
        bpp_loss = torch.mean(torch.square(z_hat))
        
        # 3. 总损失：率失真权衡（lambda可调整）
        lambda_ = 0.001  # 率失真权衡系数，越小越关注重建质量
        total_loss = recon_loss + lambda_ * bpp_loss
        
        # 返回损失字典，方便训练时监控
        return {
            "loss": total_loss,          # 总损失（用于反向传播）
            "recon_loss": recon_loss,    # 重建总损失
            "perceptual_loss": perceptual_loss,  # 感知损失
            "mse_loss": mse_loss,        # 像素损失
            "bpp": bpp_loss              # 率损失（比特率）
        }

    def forward(self, x):
        """完整前向传播（训练时用）：编码→解码→返回重建图像"""
        z_hat = self.encode(x)
        y_hat = self.decode(z_hat)
        return y_hat
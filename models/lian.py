import torch
import torch.nn as nn

from models.encoders.edsr import EDSREncoder
from models.encoders.rdn import RDNEncoder
from models.encoders.swinir import SwinIREncoder
from models.decoders.latent_decoder import LatentDecoder
from models.decoders.render_decoder import RenderDecoder


class LIAN(nn.Module):
    """局部隐式注意力网络 (Local Implicit Attention Network)"""
    
    def __init__(self, encoder_type='EDSR-b', latent_dim=256, render_dim=64, n_heads=8, n_layers=6):
        """
        初始化LIAN模型
        
        Args:
            encoder_type (str): 编码器类型，可选 'EDSR-b', 'RDN', 'SwinIR'
            latent_dim (int): 潜在空间维度
            render_dim (int): 渲染空间维度
            n_heads (int): 注意力头数
            n_layers (int): 解码器层数
        """
        super(LIAN, self).__init__()
        
        # 特征编码器
        if encoder_type == 'EDSR-b':
            self.encoder = EDSREncoder(in_channels=3, n_feats=64, n_resblocks=16)
            encoder_out_channels = 64
        elif encoder_type == 'RDN':
            self.encoder = RDNEncoder(in_channels=3, n_feats=64, growth_rate=32, num_blocks=16, num_layers=8)
            encoder_out_channels = 64
        elif encoder_type == 'SwinIR':
            self.encoder = SwinIREncoder(in_channels=3, n_feats=64, window_size=8)
            encoder_out_channels = 64
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        # LR-HD潜在解码器
        self.latent_decoder = LatentDecoder(
            in_channels=encoder_out_channels,
            latent_dim=latent_dim,
            n_heads=n_heads,
            n_layers=n_layers
        )
        
        # HR-LD渲染解码器
        self.render_decoder = RenderDecoder(
            latent_dim=latent_dim,
            render_dim=render_dim,
            n_blocks=n_layers
        )
    
    def forward(self, x, scale=None):
        """
        前向传播
        
        Args:
            x (Tensor): 输入低分辨率图像 [B, 3, h, w]
            scale (float, optional): 放大倍率，如果为None则从数据中推断
        
        Returns:
            Tensor: 重建的高分辨率图像 [B, 3, H, W]
        """
        # 如果未指定缩放因子，则默认为4
        if scale is None:
            scale = 4.0
        
        # 特征编码
        feat = self.encoder(x)  # [B, C, h, w]
        
        # LR-HD潜在解码
        latent = self.latent_decoder(feat, scale)  # [B, latent_dim, h, w]
        
        # HR-LD渲染解码
        out = self.render_decoder(latent, scale)  # [B, 3, H, W]
        
        return out 
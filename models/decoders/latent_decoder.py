import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """多头注意力模块"""
    
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, C//num_heads]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class FeedForward(nn.Module):
    """前馈网络模块"""
    
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), drop)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LatentDecoder(nn.Module):
    """低分辨率高维空间(LR-HD)潜在解码器"""
    
    def __init__(self, in_channels, latent_dim, n_heads=8, n_layers=6, mlp_ratio=4.0, dropout=0.1):
        """
        初始化LR-HD潜在解码器
        
        Args:
            in_channels (int): 输入通道数
            latent_dim (int): 潜在空间维度
            n_heads (int): 注意力头数
            n_layers (int): Transformer层数
            mlp_ratio (float): MLP隐藏层维度与输入维度的比率
            dropout (float): Dropout率
        """
        super(LatentDecoder, self).__init__()
        
        # 投影到潜在空间
        self.proj = nn.Conv2d(in_channels, latent_dim, kernel_size=1)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, 64*64, latent_dim))  # 假设特征图大小为64x64
        
        # Transformer块
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=latent_dim,
                num_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=dropout,
                attn_drop=dropout
            )
            for _ in range(n_layers)
        ])
        
        # 输出投影
        self.out_proj = nn.Conv2d(latent_dim, latent_dim, kernel_size=1)
        
        # 初始化位置编码
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x, scale=None):
        """
        前向传播
        
        Args:
            x (Tensor): 输入特征 [B, C, H, W]
            scale (float, optional): 缩放因子
        
        Returns:
            Tensor: 潜在表示 [B, latent_dim, H, W]
        """
        B, C, H, W = x.shape
        
        # 投影到潜在空间
        x = self.proj(x)  # [B, latent_dim, H, W]
        
        # 重塑为序列
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, latent_dim]
        
        # 添加位置编码
        if x.size(1) == self.pos_embed.size(1):
            x = x + self.pos_embed
        else:
            # 如果特征图大小与预定义的位置编码大小不匹配，则进行插值
            pos_embed = self.pos_embed.reshape(1, 64, 64, -1).permute(0, 3, 1, 2)
            pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, H*W, -1)
            x = x + pos_embed
        
        # 通过Transformer块
        for block in self.transformer_blocks:
            x = block(x)
        
        # 重塑回空间维度
        x = x.transpose(1, 2).reshape(B, -1, H, W)  # [B, latent_dim, H, W]
        
        # 输出投影
        x = self.out_proj(x)  # [B, latent_dim, H, W]
        
        return x 
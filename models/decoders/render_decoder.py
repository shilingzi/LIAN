import torch
import torch.nn as nn
import torch.nn.functional as F


class ImplicitModulation(nn.Module):
    """隐式调制模块"""
    
    def __init__(self, latent_dim, render_dim):
        super(ImplicitModulation, self).__init__()
        
        # 调制网络
        self.modulation = nn.Sequential(
            nn.Conv2d(latent_dim, render_dim, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(render_dim, render_dim, kernel_size=1)
        )
    
    def forward(self, x, latent):
        """
        前向传播
        
        Args:
            x (Tensor): 渲染特征 [B, render_dim, H, W]
            latent (Tensor): 潜在特征 [B, latent_dim, h, w]
        
        Returns:
            Tensor: 调制后的特征 [B, render_dim, H, W]
        """
        # 生成调制参数
        mod = self.modulation(latent)  # [B, render_dim, h, w]
        
        # 上采样调制参数到目标分辨率
        if mod.shape[2:] != x.shape[2:]:
            mod = F.interpolate(mod, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # 应用调制
        return x * mod


class RenderBlock(nn.Module):
    """渲染块"""
    
    def __init__(self, render_dim, latent_dim):
        super(RenderBlock, self).__init__()
        
        # 特征处理
        self.conv1 = nn.Conv2d(render_dim, render_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(render_dim, render_dim, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
        # 隐式调制
        self.modulation1 = ImplicitModulation(latent_dim, render_dim)
        self.modulation2 = ImplicitModulation(latent_dim, render_dim)
    
    def forward(self, x, latent):
        """
        前向传播
        
        Args:
            x (Tensor): 渲染特征 [B, render_dim, H, W]
            latent (Tensor): 潜在特征 [B, latent_dim, h, w]
        
        Returns:
            Tensor: 渲染后的特征 [B, render_dim, H, W]
        """
        # 第一个卷积+调制
        res = self.conv1(x)
        res = self.act(res)
        res = self.modulation1(res, latent)
        
        # 第二个卷积+调制
        res = self.conv2(res)
        res = self.act(res)
        res = self.modulation2(res, latent)
        
        # 残差连接
        return x + res


class RenderDecoder(nn.Module):
    """高分辨率低维空间(HR-LD)渲染解码器"""
    
    def __init__(self, latent_dim, render_dim, n_blocks=6):
        """
        初始化HR-LD渲染解码器
        
        Args:
            latent_dim (int): 潜在空间维度
            render_dim (int): 渲染空间维度
            n_blocks (int): 渲染块数量
        """
        super(RenderDecoder, self).__init__()
        
        # 初始特征生成
        self.init_feat = nn.Conv2d(latent_dim, render_dim, kernel_size=1)
        
        # 渲染块
        self.render_blocks = nn.ModuleList([
            RenderBlock(render_dim, latent_dim) for _ in range(n_blocks)
        ])
        
        # 输出层
        self.output = nn.Conv2d(render_dim, 3, kernel_size=3, padding=1)
    
    def forward(self, latent, scale):
        """
        前向传播
        
        Args:
            latent (Tensor): 潜在特征 [B, latent_dim, h, w]
            scale (float): 放大倍率
        
        Returns:
            Tensor: 重建的高分辨率图像 [B, 3, H, W]
        """
        B, C, h, w = latent.shape
        
        # 计算目标分辨率
        H, W = int(h * scale), int(w * scale)
        
        # 初始特征
        x = self.init_feat(latent)  # [B, render_dim, h, w]
        
        # 上采样到目标分辨率
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        # 通过渲染块
        for block in self.render_blocks:
            x = block(x, latent)
        
        # 输出
        x = self.output(x)
        
        return torch.clamp(x, 0, 1) 
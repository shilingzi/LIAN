import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """残差块"""
    
    def __init__(self, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2), bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
    
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class EDSREncoder(nn.Module):
    """EDSR-baseline编码器"""
    
    def __init__(self, in_channels=3, n_feats=64, n_resblocks=16, res_scale=1):
        """
        初始化EDSR-baseline编码器
        
        Args:
            in_channels (int): 输入通道数
            n_feats (int): 特征通道数
            n_resblocks (int): 残差块数量
            res_scale (float): 残差缩放因子
        """
        super(EDSREncoder, self).__init__()
        
        # 定义激活函数
        act = nn.ReLU(True)
        
        # 头部卷积层
        self.head = nn.Conv2d(in_channels, n_feats, kernel_size=3, padding=1)
        
        # 残差块
        m_body = [
            ResBlock(n_feats, kernel_size=3, res_scale=res_scale, act=act) \
            for _ in range(n_resblocks)
        ]
        m_body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1))
        
        self.body = nn.Sequential(*m_body)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (Tensor): 输入张量 [B, 3, H, W]
        
        Returns:
            Tensor: 特征张量 [B, n_feats, H, W]
        """
        # 提取特征
        x = self.head(x)
        
        # 残差学习
        res = self.body(x)
        res += x
        
        return res 
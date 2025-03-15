import torch
import torch.nn as nn


class RDB_Conv(nn.Module):
    """密集连接的卷积块"""
    
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(RDB_Conv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size, padding=(kernel_size-1)//2, stride=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    """残差密集块 (Residual Dense Block)"""
    
    def __init__(self, in_channels, growth_rate, num_layers, kernel_size=3):
        super(RDB, self).__init__()
        
        modules = []
        for i in range(num_layers):
            modules.append(RDB_Conv(in_channels + i * growth_rate, growth_rate, kernel_size))
        
        self.dense_layers = nn.Sequential(*modules)
        
        # 局部特征融合
        self.lff = nn.Conv2d(in_channels + num_layers * growth_rate, in_channels, kernel_size=1, padding=0, stride=1)
    
    def forward(self, x):
        out = self.dense_layers(x)
        out = self.lff(out)
        return out + x


class RDNEncoder(nn.Module):
    """RDN (Residual Dense Network) 编码器"""
    
    def __init__(self, in_channels=3, n_feats=64, growth_rate=32, num_blocks=16, num_layers=8):
        """
        初始化RDN编码器
        
        Args:
            in_channels (int): 输入通道数
            n_feats (int): 特征通道数
            growth_rate (int): 每个卷积层的增长率
            num_blocks (int): RDB块的数量
            num_layers (int): 每个RDB块中的卷积层数量
        """
        super(RDNEncoder, self).__init__()
        
        # 浅层特征提取
        self.sfe1 = nn.Conv2d(in_channels, n_feats, kernel_size=3, padding=1, stride=1)
        self.sfe2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, stride=1)
        
        # 残差密集块
        self.rdbs = nn.ModuleList()
        for i in range(num_blocks):
            self.rdbs.append(RDB(n_feats, growth_rate, num_layers))
        
        # 全局特征融合
        self.gff = nn.Sequential(
            nn.Conv2d(num_blocks * n_feats, n_feats, kernel_size=1, padding=0, stride=1),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, stride=1)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (Tensor): 输入张量 [B, 3, H, W]
        
        Returns:
            Tensor: 特征张量 [B, n_feats, H, W]
        """
        # 浅层特征提取
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)
        
        # RDB特征提取
        rdb_outs = []
        x = sfe2
        for rdb in self.rdbs:
            x = rdb(x)
            rdb_outs.append(x)
        
        # 全局特征融合
        x = torch.cat(rdb_outs, 1)
        x = self.gff(x)
        
        # 残差学习
        x = x + sfe1
        
        return x 
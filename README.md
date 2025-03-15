# LIAN: 基于局部隐式注意力网络的任意尺度超分辨率系统

本项目实现了一个基于局部隐式注意力网络的任意尺度超分辨率系统，可以将低分辨率图像放大到任意尺度，同时保持图像的高质量和细节。

## 特点

- 支持任意尺度的超分辨率
- 多种特征编码器选择（EDSR-b、RDN、SwinIR）
- 基于注意力机制的特征解码
- 高效的隐式神经表示

## 环境要求

- Python 3.11+
- PyTorch 2.0+
- CUDA 11.8+（用于 GPU 训练）

## 安装

1. 克隆仓库：
```bash
git clone <repository_url>
cd lian-sr
```

2. 创建并激活 conda 环境：
```bash
conda create -n lian python=3.11
conda activate lian
```

3. 安装依赖：
```bash
# 对于 GPU 训练（推荐）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt
```

## 数据集准备

1. 下载 DIV2K 数据集
2. 组织数据集结构如下：
```
data/datasets/
├── DIV2K/
│   ├── train/
│   │   └── HR/  # 高分辨率训练图像
│   └── val/
│       └── HR/  # 验证集图像
├── Set5/        # 测试集
└── Set14/       # 测试集
```

## 训练

1. 修改配置文件 `config/train_config.yaml`
2. 开始训练：
```bash
python train.py --config config/train_config.yaml
```

## 测试

使用预训练模型进行测试：
```bash
python test.py --config config/test_config.yaml --checkpoint path/to/model.pth
```

## 推理

对单张图像进行超分辨率处理：
```bash
python inference.py --input input.png --output output.png --scale 4 --config config/test_config.yaml --checkpoint path/to/model.pth
```

## 项目结构

```
.
├── config/                 # 配置文件
├── data/                  # 数据集和数据加载
├── models/                # 模型定义
│   ├── encoders/         # 特征编码器
│   └── decoders/         # 特征解码器
├── utils/                # 工具函数
├── train.py              # 训练脚本
├── test.py               # 测试脚本
└── inference.py          # 推理脚本
```

## 引用

如果您使用了本项目的代码，请引用以下论文：
```
@article{your-paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。 
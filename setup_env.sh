#!/bin/bash

# 显示每条命令的执行
set -x

# 检查是否安装了CUDA
if ! command -v nvcc &> /dev/null; then
    echo "警告: 未检测到CUDA。请确保已安装NVIDIA驱动和CUDA工具包。"
    echo "您可以使用 nvidia-smi 命令检查GPU状态。"
fi

# 检查是否安装了conda
if ! command -v conda &> /dev/null; then
    echo "未检测到conda，正在安装Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    
    # 添加conda到环境变量
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
fi

# 创建新的conda环境
echo "创建新的conda环境: lian..."
conda create -y -n lian python=3.11

# 激活环境
source ~/miniconda/bin/activate lian

# 安装PyTorch和CUDA
echo "安装PyTorch和CUDA..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
echo "安装项目依赖..."
pip install -r requirements.txt

# 安装其他必要的包
pip install tensorboard  # 用于训练监控
pip install matplotlib  # 用于可视化
pip install opencv-python  # 用于图像处理
pip install timm  # 用于SwinIR模型

# 创建必要的目录
echo "创建项目目录..."
mkdir -p data/datasets/DIV2K/{train/HR,val/HR}
mkdir -p logs/{tensorboard,checkpoints}

# 检查CUDA是否可用
echo "检查CUDA可用性..."
python - << EOF
import torch
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
EOF

echo "环境配置完成！"
echo "请使用 'conda activate lian' 激活环境"
echo "然后运行 'python train.py --config config/train_config.yaml' 开始训练" 
import os
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from models.lian import LIAN
from utils.config import load_config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LIAN Inference')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--scale', type=float, default=4.0, help='Upscale factor')
    return parser.parse_args()


def load_image(path):
    """加载图像"""
    img = Image.open(path).convert('RGB')
    return img


def preprocess(img):
    """预处理图像"""
    # 转换为张量
    to_tensor = transforms.ToTensor()
    tensor = to_tensor(img).unsqueeze(0)  # 添加批次维度
    return tensor


def postprocess(tensor):
    """后处理图像"""
    # 确保值在[0, 1]范围内
    tensor = torch.clamp(tensor, 0, 1)
    
    # 转换为PIL图像
    if tensor.ndim == 4:
        tensor = tensor[0]
    
    img = TF.to_pil_image(tensor)
    return img


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    model = LIAN(
        encoder_type=config.model.encoder,
        latent_dim=config.model.latent_dim,
        render_dim=config.model.render_dim,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers
    ).to(device)
    
    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # 加载图像
    img = load_image(args.input)
    print(f"Loaded image: {args.input}, size: {img.size}")
    
    # 预处理
    tensor = preprocess(img).to(device)
    
    # 推理
    with torch.no_grad():
        output = model(tensor, args.scale)
    
    # 后处理
    output_img = postprocess(output.cpu())
    
    # 保存结果
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    output_img.save(args.output)
    print(f"Saved result to: {args.output}, size: {output_img.size}")


if __name__ == '__main__':
    main()
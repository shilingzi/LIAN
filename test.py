import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from models.lian import LIAN
from data.datasets import get_dataloader
from utils.config import load_config
from utils.metrics import calculate_psnr, calculate_ssim


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Test LIAN model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--save_results', action='store_true', help='Save SR results')
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save results')
    return parser.parse_args()


def save_image(tensor, path):
    """保存图像"""
    # 转换为PIL图像
    if tensor.ndim == 4:
        tensor = tensor[0]
    
    # 确保值在[0, 1]范围内
    tensor = torch.clamp(tensor, 0, 1)
    
    # 转换为PIL图像
    img = TF.to_pil_image(tensor)
    
    # 保存图像
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def test_on_dataset(model, dataloader, device, scales, save_results=False, results_dir=None):
    """在数据集上测试模型"""
    model.eval()
    
    # 为每个缩放因子初始化指标
    metrics = {scale: {'psnr': [], 'ssim': []} for scale in scales}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Testing'):
            # 获取数据
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            names = batch['name']
            
            # 对每个缩放因子进行测试
            for scale in scales:
                # 前向传播
                sr = model(lr, scale)
                
                # 计算评估指标
                psnr = calculate_psnr(sr, hr)
                ssim = calculate_ssim(sr, hr)
                
                # 更新指标
                metrics[scale]['psnr'].append(psnr)
                metrics[scale]['ssim'].append(ssim)
                
                # 保存结果
                if save_results and results_dir:
                    for i, name in enumerate(names):
                        save_path = os.path.join(results_dir, f'x{scale}', name)
                        save_image(sr[i], save_path)
    
    # 计算平均指标
    for scale in scales:
        metrics[scale]['psnr'] = np.mean(metrics[scale]['psnr'])
        metrics[scale]['ssim'] = np.mean(metrics[scale]['ssim'])
    
    return metrics


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
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # 创建结果目录
    if args.save_results:
        for scale in config.test.scales:
            os.makedirs(os.path.join(args.results_dir, f'x{scale}'), exist_ok=True)
    
    # 测试每个数据集
    for dataset_name in config.data.test_datasets:
        print(f"Testing on {dataset_name}...")
        
        # 创建数据加载器
        config.data.val_dataset = dataset_name
        config.data.val_dir = config.data.test_dirs[dataset_name]
        dataloader = get_dataloader(config, is_train=False)
        
        # 测试模型
        metrics = test_on_dataset(
            model,
            dataloader,
            device,
            config.test.scales,
            save_results=args.save_results,
            results_dir=args.results_dir
        )
        
        # 打印结果
        print(f"Results on {dataset_name}:")
        for scale in config.test.scales:
            print(f"  Scale x{scale}:")
            print(f"    PSNR: {metrics[scale]['psnr']:.4f} dB")
            print(f"    SSIM: {metrics[scale]['ssim']:.4f}")


if __name__ == '__main__':
    main() 
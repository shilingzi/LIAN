import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.lian import LIAN
from data.datasets import get_dataloader
from utils.config import load_config, save_config
from utils.metrics import calculate_psnr, calculate_ssim


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train LIAN model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_optimizer(model, config):
    """获取优化器"""
    if config.optimizer.type == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.train.lr,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
            weight_decay=config.train.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer.type}")
    
    return optimizer


def get_scheduler(optimizer, config):
    """获取学习率调度器"""
    if config.scheduler.type == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.scheduler.T_max,
            eta_min=config.train.min_lr
        )
    elif config.scheduler.type == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.scheduler.milestones,
            gamma=config.scheduler.gamma
        )
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler.type}")
    
    return scheduler


def get_loss_fn(config):
    """获取损失函数"""
    if config.loss.type == 'L1':
        loss_fn = nn.L1Loss()
    elif config.loss.type == 'MSE':
        loss_fn = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss: {config.loss.type}")
    
    return loss_fn


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    with tqdm(dataloader, desc='Training') as pbar:
        for batch in pbar:
            # 获取数据
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            scale = batch['scale'].to(device)
            
            # 前向传播
            optimizer.zero_grad()
            sr = model(lr, scale[0].item())
            
            # 计算损失
            loss = loss_fn(sr, hr)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 更新统计信息
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, loss_fn, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    
    with torch.no_grad():
        with tqdm(dataloader, desc='Validation') as pbar:
            for batch in pbar:
                # 获取数据
                lr = batch['lr'].to(device)
                hr = batch['hr'].to(device)
                scale = batch['scale'].to(device)
                
                # 前向传播
                sr = model(lr, scale[0].item())
                
                # 计算损失
                loss = loss_fn(sr, hr)
                
                # 计算评估指标
                psnr = calculate_psnr(sr, hr)
                ssim = calculate_ssim(sr, hr)
                
                # 更新统计信息
                total_loss += loss.item()
                total_psnr += psnr
                total_ssim += ssim
                pbar.set_postfix({
                    'loss': loss.item(),
                    'psnr': psnr,
                    'ssim': ssim
                })
    
    return {
        'loss': total_loss / len(dataloader),
        'psnr': total_psnr / len(dataloader),
        'ssim': total_ssim / len(dataloader)
    }


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建日志目录
    os.makedirs(config.log.tensorboard_dir, exist_ok=True)
    os.makedirs(config.log.save_dir, exist_ok=True)
    
    # 保存配置
    save_config(config, os.path.join(config.log.save_dir, 'config.yaml'))
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建数据加载器
    train_loader = get_dataloader(config, is_train=True)
    val_loader = get_dataloader(config, is_train=False)
    
    # 创建模型
    model = LIAN(
        encoder_type=config.model.encoder,
        latent_dim=config.model.latent_dim,
        render_dim=config.model.render_dim,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers
    ).to(device)
    
    # 创建优化器和学习率调度器
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    # 创建损失函数
    loss_fn = get_loss_fn(config)
    
    # 创建TensorBoard写入器
    writer = SummaryWriter(config.log.tensorboard_dir)
    
    # 加载检查点（如果有）
    start_epoch = 0
    best_psnr = 0
    if config.train.resume and config.train.checkpoint:
        checkpoint = torch.load(config.train.checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint['best_psnr']
        print(f"Resuming from epoch {start_epoch}, best PSNR: {best_psnr:.4f}")
    
    # 训练循环
    for epoch in range(start_epoch, config.train.epochs):
        print(f"Epoch {epoch+1}/{config.train.epochs}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录训练损失
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # 验证
        if (epoch + 1) % config.train.val_every == 0:
            val_metrics = validate(model, val_loader, loss_fn, device)
            
            # 记录验证指标
            writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            writer.add_scalar('PSNR/val', val_metrics['psnr'], epoch)
            writer.add_scalar('SSIM/val', val_metrics['ssim'], epoch)
            
            print(f"Validation: Loss: {val_metrics['loss']:.4f}, PSNR: {val_metrics['psnr']:.4f}, SSIM: {val_metrics['ssim']:.4f}")
            
            # 保存最佳模型
            if val_metrics['psnr'] > best_psnr:
                best_psnr = val_metrics['psnr']
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_psnr': best_psnr
                }, os.path.join(config.log.save_dir, 'best_model.pth'))
                print(f"Saved best model with PSNR: {best_psnr:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % config.train.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_psnr': best_psnr
            }, os.path.join(config.log.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # 保存最终模型
    torch.save({
        'epoch': config.train.epochs - 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_psnr': best_psnr
    }, os.path.join(config.log.save_dir, 'final_model.pth'))
    
    writer.close()
    print("Training completed!")


if __name__ == '__main__':
    main() 
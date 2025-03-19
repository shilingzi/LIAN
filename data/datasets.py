import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class SRDataset(Dataset):
    """超分辨率数据集基类"""
    
    def __init__(self, root_dir, patch_size=48, scale=None, is_train=True, augment=True):
        """
        初始化超分辨率数据集
        
        Args:
            root_dir (str): 数据集根目录
            patch_size (int): 训练时LR图像的裁剪大小
            scale (float): 放大倍率，如果为None则为任意尺度
            is_train (bool): 是否为训练集
            augment (bool): 是否进行数据增强
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.scale = scale
        self.is_train = is_train
        self.augment = augment and is_train
        
        # 获取图像文件列表
        self.hr_files = self._get_image_files()
        
        # 图像转换
        self.to_tensor = transforms.ToTensor()
        
    def _get_image_files(self):
        """获取HR图像文件列表，子类需要实现此方法"""
        raise NotImplementedError
    
    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        # 读取HR图像
        hr_path = self.hr_files[idx]
        hr_img = Image.open(hr_path).convert('RGB')
        
        # 随机裁剪
        if self.is_train:
            # 确保HR图像足够大以供裁剪
            h, w = hr_img.size
            if h < self.patch_size * 4 or w < self.patch_size * 4:
                # 如果图像太小，进行上采样
                hr_img = hr_img.resize((max(h, self.patch_size * 4), 
                                        max(w, self.patch_size * 4)), 
                                       Image.BICUBIC)
            
            # 随机裁剪HR图像
            h, w = hr_img.size
            x = random.randint(0, h - self.patch_size * 4)
            y = random.randint(0, w - self.patch_size * 4)
            hr_img = hr_img.crop((x, y, x + self.patch_size * 4, y + self.patch_size * 4))
            
            # 生成LR图像
            lr_img = hr_img.resize((self.patch_size, self.patch_size), Image.BICUBIC)
            
            # 数据增强
            if self.augment:
                # 随机水平翻转
                if random.random() < 0.5:
                    hr_img = TF.hflip(hr_img)
                    lr_img = TF.hflip(lr_img)
                
                # 随机垂直翻转
                if random.random() < 0.5:
                    hr_img = TF.vflip(hr_img)
                    lr_img = TF.vflip(lr_img)
                
                # 随机旋转90度
                if random.random() < 0.5:
                    angle = random.choice([90, 180, 270])
                    hr_img = TF.rotate(hr_img, angle)
                    lr_img = TF.rotate(lr_img, angle)
        else:
            # 测试时，直接调整LR图像大小
            if self.scale is not None:
                w, h = hr_img.size
                lr_img = hr_img.resize((int(w // self.scale), int(h // self.scale)), Image.BICUBIC)
            else:
                # 任意尺度超分，随机选择一个缩放因子
                w, h = hr_img.size
                random_scale = random.uniform(2, 4) if self.is_train else 4.0
                lr_img = hr_img.resize((int(w // random_scale), int(h // random_scale)), Image.BICUBIC)
        
        # 转换为张量
        hr_tensor = self.to_tensor(hr_img)
        lr_tensor = self.to_tensor(lr_img)
        
        # 计算实际缩放因子
        if self.scale is None:
            actual_scale = hr_tensor.shape[-1] / lr_tensor.shape[-1]
        else:
            actual_scale = self.scale
            
        return {
            'lr': lr_tensor,
            'hr': hr_tensor,
            'scale': torch.tensor(actual_scale, dtype=torch.float32),
            'name': os.path.basename(hr_path)
        }


class DIV2KDataset(SRDataset):
    """DIV2K数据集"""
    
    def _get_image_files(self):
        """获取DIV2K数据集的HR图像文件列表"""
        hr_dir = os.path.join(self.root_dir, 'HR')
        if not os.path.exists(hr_dir):
            raise ValueError(f"HR directory not found: {hr_dir}")
        
        hr_files = []
        for filename in sorted(os.listdir(hr_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                hr_files.append(os.path.join(hr_dir, filename))
        
        return hr_files


class BenchmarkDataset(SRDataset):
    """测试基准数据集（Set5, Set14等）"""
    
    def _get_image_files(self):
        """获取测试数据集的HR图像文件列表"""
        hr_files = []
        for filename in sorted(os.listdir(self.root_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                hr_files.append(os.path.join(self.root_dir, filename))
        
        return hr_files


def get_dataloader(config, is_train=True):
    """
    获取数据加载器
    
    Args:
        config: 配置对象
        is_train: 是否为训练模式
    
    Returns:
        数据加载器
    """
    if is_train:
        # 训练数据集
        dataset_name = config.data.train_dataset
        dataset_dir = config.data.train_dir
        batch_size = config.data.batch_size
        shuffle = True
        
        if dataset_name == 'DIV2K':
            dataset = DIV2KDataset(
                root_dir=dataset_dir,
                patch_size=config.data.patch_size,
                scale=None,  # 任意尺度
                is_train=True,
                augment=config.data.augment
            )
        else:
            raise ValueError(f"Unsupported training dataset: {dataset_name}")
    else:
        # 验证/测试数据集
        dataset_name = config.data.val_dataset if hasattr(config.data, 'val_dataset') else config.data.test_datasets[0]
        dataset_dir = config.data.val_dir if hasattr(config.data, 'val_dir') else config.data.test_dirs[dataset_name]
        batch_size = 1  # 测试时总是使用batch_size=1
        shuffle = False
        
        if dataset_name == 'DIV2K100':
            dataset = DIV2KDataset(
                root_dir=dataset_dir,
                patch_size=None,
                scale=None,  # 任意尺度
                is_train=False,
                augment=False
            )
        else:
            dataset = BenchmarkDataset(
                root_dir=dataset_dir,
                patch_size=None,
                scale=None,  # 任意尺度
                is_train=False,
                augment=False
            )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
        pin_memory=getattr(config.data, 'pin_memory', True),  # 默认启用pin_memory
        prefetch_factor=getattr(config.data, 'prefetch_factor', 2),  # 默认预加载因子为2
        persistent_workers=True,  # 保持worker进程存活以减少重启开销
        drop_last=is_train  # 训练时丢弃不完整的批次
    )
    
    return dataloader 
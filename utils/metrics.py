import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr(img1, img2, crop_border=0, test_y_channel=True):
    """
    计算PSNR (Peak Signal-to-Noise Ratio)
    
    Args:
        img1 (Tensor or ndarray): 图像1
        img2 (Tensor or ndarray): 图像2
        crop_border (int): 裁剪边界像素数
        test_y_channel (bool): 是否只在Y通道上计算
    
    Returns:
        float: PSNR值
    """
    # 确保输入是numpy数组
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # 如果是4D张量，取第一个样本
    if img1.ndim == 4:
        img1 = img1[0]
    if img2.ndim == 4:
        img2 = img2[0]
    
    # 转换为[0, 255]范围
    if img1.max() <= 1.0:
        img1 = img1 * 255.0
    if img2.max() <= 1.0:
        img2 = img2 * 255.0
    
    # 转换为HWC格式
    if img1.shape[0] == 3:
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.shape[0] == 3:
        img2 = np.transpose(img2, (1, 2, 0))
    
    # 裁剪边界
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    # 转换为Y通道 (YCbCr)
    if test_y_channel and img1.shape[-1] == 3:
        img1_y = rgb2ycbcr(img1)[:, :, 0]
        img2_y = rgb2ycbcr(img2)[:, :, 0]
        return peak_signal_noise_ratio(img1_y, img2_y)
    else:
        return peak_signal_noise_ratio(img1, img2)


def calculate_ssim(img1, img2, crop_border=0, test_y_channel=True):
    """
    计算SSIM (Structural Similarity Index)
    
    Args:
        img1 (Tensor or ndarray): 图像1
        img2 (Tensor or ndarray): 图像2
        crop_border (int): 裁剪边界像素数
        test_y_channel (bool): 是否只在Y通道上计算
    
    Returns:
        float: SSIM值
    """
    # 确保输入是numpy数组
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # 如果是4D张量，取第一个样本
    if img1.ndim == 4:
        img1 = img1[0]
    if img2.ndim == 4:
        img2 = img2[0]
    
    # 转换为[0, 255]范围
    if img1.max() <= 1.0:
        img1 = img1 * 255.0
    if img2.max() <= 1.0:
        img2 = img2 * 255.0
    
    # 转换为HWC格式
    if img1.shape[0] == 3:
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.shape[0] == 3:
        img2 = np.transpose(img2, (1, 2, 0))
    
    # 裁剪边界
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    # 转换为Y通道 (YCbCr)
    if test_y_channel and img1.shape[-1] == 3:
        img1_y = rgb2ycbcr(img1)[:, :, 0]
        img2_y = rgb2ycbcr(img2)[:, :, 0]
        return structural_similarity(img1_y, img2_y)
    else:
        return structural_similarity(img1, img2, multichannel=True)


def rgb2ycbcr(img):
    """
    RGB转YCbCr
    
    Args:
        img (ndarray): RGB图像，范围[0, 255]
    
    Returns:
        ndarray: YCbCr图像
    """
    y = 16. + (65.481 * img[:, :, 0] + 128.553 * img[:, :, 1] + 24.966 * img[:, :, 2]) / 255.
    cb = 128. + (-37.797 * img[:, :, 0] - 74.203 * img[:, :, 1] + 112.0 * img[:, :, 2]) / 255.
    cr = 128. + (112.0 * img[:, :, 0] - 93.786 * img[:, :, 1] - 18.214 * img[:, :, 2]) / 255.
    return np.stack((y, cb, cr), axis=2) 
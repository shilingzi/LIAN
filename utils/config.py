import os
import yaml
from typing import Dict, Any
from types import SimpleNamespace

def dict_to_namespace(d: Dict[str, Any]) -> SimpleNamespace:
    """
    将字典转换为命名空间对象
    Args:
        d: 输入字典
    Returns:
        命名空间对象
    """
    namespace = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

def load_config(config_path: str) -> SimpleNamespace:
    """
    加载YAML配置文件
    Args:
        config_path: 配置文件路径
    Returns:
        配置命名空间对象
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config_dict = yaml.safe_load(f)
            return dict_to_namespace(config_dict)
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误: {e}")

def save_config(config: SimpleNamespace, save_path: str) -> None:
    """
    保存配置到YAML文件
    Args:
        config: 配置命名空间对象
        save_path: 保存路径
    """
    def namespace_to_dict(ns: SimpleNamespace) -> Dict[str, Any]:
        """将命名空间对象转换回字典"""
        result = {}
        for key, value in ns.__dict__.items():
            if isinstance(value, SimpleNamespace):
                result[key] = namespace_to_dict(value)
            else:
                result[key] = value
        return result
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    config_dict = namespace_to_dict(config)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True) 
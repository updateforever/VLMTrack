# -*- coding: utf-8 -*-
"""
SOIBench/vlms/model_adapters/__init__.py
模型适配器包
"""

from .base import ModelAdapter
from .qwen3vl import Qwen3VLAdapter
from .glm46v import GLM46VAdapter
from .deepseekvl import DeepSeekVLAdapter

__all__ = [
    'ModelAdapter',
    'Qwen3VLAdapter',
    'GLM46VAdapter',
    'DeepSeekVLAdapter',
]

# 模型注册表
ADAPTER_REGISTRY = {
    'qwen3vl': Qwen3VLAdapter,
    'glm46v': GLM46VAdapter,
    'deepseekvl': DeepSeekVLAdapter,
}


def get_adapter(model_name: str) -> ModelAdapter:
    """
    根据模型名称获取适配器
    
    参数:
        model_name: 模型名称 (qwen3vl, glm46v, deepseekvl)
    
    返回:
        对应的适配器类
    """
    if model_name not in ADAPTER_REGISTRY:
        raise ValueError(
            f"未知的模型: {model_name}. "
            f"支持的模型: {list(ADAPTER_REGISTRY.keys())}"
        )
    return ADAPTER_REGISTRY[model_name]


def register_adapter(model_name: str, adapter_class):
    """
    注册新的模型适配器
    
    参数:
        model_name: 模型名称
        adapter_class: 适配器类（需继承 ModelAdapter）
    
    示例:
        >>> class MyVLMAdapter(ModelAdapter):
        ...     pass
        >>> register_adapter('myvlm', MyVLMAdapter)
    """
    if not issubclass(adapter_class, ModelAdapter):
        raise TypeError(f"{adapter_class} 必须继承 ModelAdapter")
    ADAPTER_REGISTRY[model_name] = adapter_class
    print(f"✅ 已注册模型适配器: {model_name}")

# -*- coding: utf-8 -*-
"""
SOIBench/vlms/model_adapters/base.py
模型适配器基类
定义所有 VLM 模型需要实现的接口
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any


class ModelAdapter(ABC):
    """
    VLM 模型适配器基类
    
    所有新增的 VLM 模型都需要继承此类并实现以下方法：
    1. build_prompt() - 构造模型特定的 prompt
    2. parse_response() - 解析模型输出的 bbox
    3. create_engine() - 创建推理引擎
    """
    
    @abstractmethod
    def build_prompt(self, desc_parts: List[str]) -> str:
        """
        构造模型特定的 prompt
        
        参数:
            desc_parts: 描述文本列表，已经过标点处理
                       例如: ["a red car,", "with black windows,", "driving on the road.", "in the city center."]
        
        返回:
            模型特定格式的 prompt 字符串
        
        示例:
            Qwen3VL: "a red car, with black windows, driving on the road. in the city center."
            GLM-4.6V: "Please pinpoint the bounding box in the image as per the given description: a red car, ..."
            DeepSeek-VL2: "<|ref|>a red car, ...<|/ref|>."
        """
        pass
    
    @abstractmethod
    def parse_response(self, response: str, img_width: int, img_height: int) -> List[List[float]]:
        """
        解析模型输出，提取 bbox
        
        参数:
            response: 模型原始输出文本
            img_width: 图像宽度（像素）
            img_height: 图像高度（像素）
        
        返回:
            bbox 列表，每个 bbox 为 [x1, y1, x2, y2] 像素坐标
            如果解析失败，返回空列表 []
        
        注意:
            - 需要处理模型特定的输出格式
            - 需要将归一化坐标转换为像素坐标
            - 需要确保坐标在图像范围内
        """
        pass
    
    @abstractmethod
    def create_engine(self, args) -> Any:
        """
        创建推理引擎
        
        参数:
            args: 命令行参数对象，包含:
                - mode: 'local' 或 'api'
                - model_path: 本地模型路径 (mode=local 时)
                - api_key: API 密钥 (mode=api 时)
                - api_model_name: API 模型名称
                - api_base_url: API 基础 URL
                - api_temperature: 温度参数
                - api_max_tokens: 最大 token 数
                - api_retries: 重试次数
        
        返回:
            推理引擎对象，需要有 chat(image_path, prompt) 方法
        
        示例:
            if args.mode == 'local':
                return MyVLMLocalEngine(args.model_path)
            else:
                return MyVLMAPIEngine(
                    api_key=args.api_key,
                    model_name=args.api_model_name,
                    ...
                )
        """
        pass
    
    def get_default_model_path(self) -> str:
        """
        获取默认的本地模型路径
        
        返回:
            默认模型路径字符串
        
        注意:
            子类可以重写此方法提供默认路径
        """
        return None
    
    def get_default_api_model_name(self) -> str:
        """
        获取默认的 API 模型名称
        
        返回:
            默认 API 模型名称字符串
        
        注意:
            子类可以重写此方法提供默认值
        """
        return None
    
    def get_default_api_base_url(self) -> str:
        """
        获取默认的 API Base URL
        
        返回:
            默认 API Base URL 字符串
        
        注意:
            子类可以重写此方法提供默认值
        """
        return "https://api.siliconflow.cn/v1"
    
    def preprocess_description(self, desc_parts: List[str]) -> List[str]:
        """
        预处理描述文本（可选）
        
        参数:
            desc_parts: 原始描述文本列表
        
        返回:
            处理后的描述文本列表
        
        注意:
            默认实现直接返回原始文本
            子类可以重写此方法进行特殊处理
        """
        return desc_parts
    
    def postprocess_bboxes(self, bboxes: List[List[float]], img_width: int, img_height: int) -> List[List[float]]:
        """
        后处理 bbox（可选）
        
        参数:
            bboxes: 解析出的 bbox 列表
            img_width: 图像宽度
            img_height: 图像高度
        
        返回:
            处理后的 bbox 列表
        
        注意:
            默认实现直接返回原始 bbox
            子类可以重写此方法进行特殊处理（如过滤、排序等）
        """
        return bboxes

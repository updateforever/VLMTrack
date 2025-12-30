# -*- coding: utf-8 -*-
"""
SOIBench/vlms/model_adapters/deepseekvl_adapter.py
DeepSeek-VL2 模型适配器
"""

from typing import List
from .base import ModelAdapter


class DeepSeekVLAdapter(ModelAdapter):
    """DeepSeek-VL2 模型适配器"""
    
    def build_prompt(self, desc_parts: List[str]) -> str:
        """
        DeepSeek-VL2 的 prompt 格式：
        "<|ref|>{description}<|/ref|>."
        """
        description = " ".join(desc_parts).strip()
        return f"<|ref|>{description}<|/ref|>."
    
    def parse_response(self, response: str, img_width: int, img_height: int) -> List[List[float]]:
        """
        解析 DeepSeek-VL2 的输出
        使用 deepseekvl_infer.py 中的 parse_deepseekvl_bbox 函数
        """
        from deepseekvl_infer import parse_deepseekvl_bbox
        return parse_deepseekvl_bbox(response, img_width, img_height)
    
    def create_engine(self, args):
        """创建 DeepSeek-VL2 推理引擎"""
        if args.mode == 'local':
            from deepseekvl_infer import DeepSeekVLLocalEngine
            return DeepSeekVLLocalEngine(args.model_path)
        else:
            from deepseekvl_infer import DeepSeekVLAPIEngine
            return DeepSeekVLAPIEngine(
                api_key=args.api_key,
                api_base_url=args.api_base_url,
                model_name=args.api_model_name,
                temperature=args.api_temperature,
                max_tokens=args.api_max_tokens,
                retries=args.api_retries,
            )
    
    def get_default_model_path(self) -> str:
        return "/home/member/data1/MODEL_WEIGHTS_PUBLIC/deepseek-vl2-small/"
    
    def get_default_api_model_name(self) -> str:
        return "deepseek-ai/deepseek-vl2"
    
    def get_default_api_base_url(self) -> str:
        return "https://api.siliconflow.cn/v1"

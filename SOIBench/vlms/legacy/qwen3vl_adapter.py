# -*- coding: utf-8 -*-
"""
SOIBench/vlms/model_adapters/qwen3vl_adapter.py
Qwen3VL 模型适配器
"""

import json
import re
from typing import List
from .base import ModelAdapter


class Qwen3VLAdapter(ModelAdapter):
    """Qwen3VL 模型适配器"""
    
    def build_prompt(self, desc_parts: List[str]) -> str:
        """
        Qwen3VL 的 prompt 格式：直接拼接描述文本
        """
        return " ".join(desc_parts).strip()
    
    def parse_response(self, response: str, img_width: int, img_height: int) -> List[List[float]]:
        """
        解析 Qwen3VL 的输出
        支持多种格式:
        1) JSON: {"bbox_2d":[...]} 或 [{"bbox_2d":[...]}]
        2) JSON: [x1,y1,x2,y2]
        3) 文本中包含多个 [x1,y1,x2,y2]
        4) 兼容 0-1, 0-1000, 像素坐标
        """
        raw = response or ""
        t = self._strip_code_fence(raw)
        if not t:
            return []
        
        bboxes = []
        
        # 尝试解析 JSON
        try:
            data = json.loads(t)
            
            if isinstance(data, dict):
                # 单个 bbox: {"bbox_2d": [x1,y1,x2,y2]}
                if "bbox_2d" in data:
                    b = self._safe_float_list(data["bbox_2d"])
                    if b:
                        bboxes.append(b)
                        
            elif isinstance(data, list):
                # 多个 bbox: [{"bbox_2d":[...]}, ...]
                for item in data:
                    if isinstance(item, dict) and "bbox_2d" in item:
                        b = self._safe_float_list(item["bbox_2d"])
                        if b:
                            bboxes.append(b)
                    # 或直接是坐标列表: [[x1,y1,x2,y2], ...]
                    elif isinstance(item, (list, tuple)):
                        b = self._safe_float_list(item)
                        if b:
                            bboxes.append(b)
                
                # 或单个 bbox: [x1,y1,x2,y2]
                if not bboxes:
                    b = self._safe_float_list(data)
                    if b:
                        bboxes.append(b)
                        
        except Exception:
            pass
        
        # 如果 JSON 解析失败，尝试正则提取
        if not bboxes:
            pattern = r'\[?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]?'
            matches = re.findall(pattern, raw)
            for m in matches:
                try:
                    b = [float(x) for x in m]
                    if len(b) == 4:
                        bboxes.append(b)
                except Exception:
                    continue
        
        # 转换为像素坐标并裁剪
        result = []
        for bbox in bboxes:
            pixel_bbox = self._convert_to_pixel_bbox(bbox, img_width, img_height)
            fixed_bbox = self._fix_and_clip_bbox(pixel_bbox, img_width, img_height)
            result.append(fixed_bbox)
        
        return result
    
    def create_engine(self, args):
        """创建 Qwen3VL 推理引擎"""
        if args.mode == 'local':
            from qwen3vl_infer import Qwen3VLLocalEngine
            return Qwen3VLLocalEngine(args.model_path)
        else:
            # API 模式返回一个包装器
            from qwen3vl_infer import qwen3vl_api_chat
            
            class Qwen3VLAPIWrapper:
                def __init__(self, args):
                    self.args = args
                
                def chat(self, image_path, prompt, max_new_tokens=None):
                    return qwen3vl_api_chat(
                        image_path=image_path,
                        prompt=prompt,
                        model_name=self.args.api_model_name,
                        base_url=self.args.api_base_url,
                        api_key=self.args.api_key or os.getenv("DASHSCOPE_API_KEY"),
                        temperature=self.args.api_temperature,
                        max_tokens=max_new_tokens or self.args.api_max_tokens,
                        retries=self.args.api_retries,
                        retry_sleep=1.0,
                    )
            
            import os
            return Qwen3VLAPIWrapper(args)
    
    def get_default_model_path(self) -> str:
        return "/home/member/data1/MODEL_WEIGHTS_PUBLIC/Qwen3-VL-32B-Instruct/"
    
    def get_default_api_model_name(self) -> str:
        return "qwen3-vl-32b-instruct"
    
    def get_default_api_base_url(self) -> str:
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    # 辅助方法
    def _strip_code_fence(self, text: str) -> str:
        """去掉 ```json / ``` 等代码块包裹"""
        if not text:
            return ""
        t = text.strip()
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
        return t.strip()
    
    def _safe_float_list(self, x):
        """尝试把输入转换成长度为 4 的 float list"""
        if isinstance(x, (list, tuple)) and len(x) == 4:
            try:
                return [float(v) for v in x]
            except Exception:
                return None
        return None
    
    def _clamp(self, v, lo, hi):
        """将值限制在 [lo, hi] 范围内"""
        return max(lo, min(hi, v))
    
    def _fix_and_clip_bbox(self, b, w, h):
        """修正 bbox 坐标顺序并裁剪到图像范围内"""
        x1, y1, x2, y2 = b
        # 确保 x1 < x2, y1 < y2
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        
        # 裁剪到图像范围
        x1 = self._clamp(x1, 0.0, float(w - 1))
        y1 = self._clamp(y1, 0.0, float(h - 1))
        x2 = self._clamp(x2, 0.0, float(w - 1))
        y2 = self._clamp(y2, 0.0, float(h - 1))
        
        # 确保 bbox 至少有 1 像素宽高
        if abs(x2 - x1) < 1.0:
            x2 = self._clamp(x1 + 1.0, 0.0, float(w - 1))
        if abs(y2 - y1) < 1.0:
            y2 = self._clamp(y1 + 1.0, 0.0, float(h - 1))
        
        return [x1, y1, x2, y2]
    
    def _convert_to_pixel_bbox(self, b, w, h):
        """
        支持三类坐标体系并统一成像素坐标
        1) 0 到 1 归一化坐标
        2) 0 到 1000 归一化坐标
        3) 像素坐标
        """
        x1, y1, x2, y2 = b
        maxv = max(x1, y1, x2, y2)
        minv = min(x1, y1, x2, y2)
        
        # 判断是 0-1 归一化
        if 0.0 <= minv and maxv <= 1.0:
            return [x1 * w, y1 * h, x2 * w, y2 * h]
        
        # 判断是 0-1000 归一化
        if 0.0 <= minv and maxv <= 1000.0:
            return [(x1 / 1000.0) * w, (y1 / 1000.0) * h, (x2 / 1000.0) * w, (y2 / 1000.0) * h]
        
        # 否则认为是像素坐标
        return [x1, y1, x2, y2]

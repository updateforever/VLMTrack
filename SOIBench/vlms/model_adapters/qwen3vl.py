# -*- coding: utf-8 -*-
"""
SOIBench/vlms/model_adapters/qwen3vl.py
Qwen3VL 模型完整实现
包含推理引擎、bbox 解析和适配器
"""

import os
import time
import base64
import json
import re
from typing import List
from .base import ModelAdapter


# ============================================================================
# 推理引擎
# ============================================================================
def _pick_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


class Qwen3VLLocalEngine:
    """Qwen3VL 本地推理引擎"""
    
    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        device_map: str = "auto",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
    ):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        
        print(f"🚀 加载 Qwen3VL 本地模型: {model_path}")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        print("✅ Qwen3VL 模型加载完成")
    
    def chat(self, image_path: str, prompt: str, max_new_tokens: int = 256) -> str:
        """单张图推理"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        # 去掉 prompt 对应的 token
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0] or ""


class Qwen3VLAPIEngine:
    """Qwen3VL API 推理引擎"""
    
    def __init__(
        self,
        api_key: str = None,
        api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name: str = "qwen3-vl-32b-instruct",
        temperature: float = 0.1,
        max_tokens: int = 256,
        retries: int = 3,
    ):
        from openai import OpenAI
        
        if api_key is None:
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
        
        self.client = OpenAI(api_key=api_key, base_url=api_base_url)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        
        print(f"🚀 初始化 Qwen3VL API 引擎")
        print(f"   模型: {model_name}")
        print(f"   Base URL: {api_base_url}")
    
    def chat(self, image_path: str, prompt: str, max_new_tokens: int = None) -> str:
        """单张图推理"""
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        for attempt in range(self.retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_new_tokens or self.max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                if attempt < self.retries - 1:
                    print(f"⚠️  API 调用失败 (尝试 {attempt + 1}/{self.retries}): {e}")
                    time.sleep(2 ** attempt)
                else:
                    print(f"❌ API 调用失败: {e}")
                    raise


# ============================================================================
# Bbox 解析
# ============================================================================

def parse_qwen3vl_bbox(response: str, img_width: int, img_height: int) -> List[List[float]]:
    """
    解析 Qwen3VL 的 bbox 输出
    支持多种格式:
    1) JSON: {"bbox_2d":[...]} 或 [{"bbox_2d":[...]}]
    2) JSON: [x1,y1,x2,y2]
    3) 文本中包含多个 [x1,y1,x2,y2]
    4) 兼容 0-1, 0-1000, 像素坐标
    """
    def strip_code_fence(text):
        if not text:
            return ""
        t = text.strip()
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
        return t.strip()
    
    def safe_float_list(x):
        if isinstance(x, (list, tuple)) and len(x) == 4:
            try:
                return [float(v) for v in x]
            except Exception:
                return None
        return None
    
    def convert_to_pixel(b, w, h):
        x1, y1, x2, y2 = b
        maxv = max(x1, y1, x2, y2)
        minv = min(x1, y1, x2, y2)
        
        if 0.0 <= minv and maxv <= 1.0:
            return [x1 * w, y1 * h, x2 * w, y2 * h]
        if 0.0 <= minv and maxv <= 1000.0:
            return [(x1 / 1000.0) * w, (y1 / 1000.0) * h, (x2 / 1000.0) * w, (y2 / 1000.0) * h]
        return [x1, y1, x2, y2]
    
    def fix_and_clip(b, w, h):
        x1, y1, x2, y2 = b
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        
        x1 = max(0.0, min(x1, float(w - 1)))
        y1 = max(0.0, min(y1, float(h - 1)))
        x2 = max(0.0, min(x2, float(w - 1)))
        y2 = max(0.0, min(y2, float(h - 1)))
        
        if abs(x2 - x1) < 1.0:
            x2 = max(0.0, min(x1 + 1.0, float(w - 1)))
        if abs(y2 - y1) < 1.0:
            y2 = max(0.0, min(y1 + 1.0, float(h - 1)))
        
        return [x1, y1, x2, y2]
    
    raw = response or ""
    t = strip_code_fence(raw)
    if not t:
        return []
    
    bboxes = []
    
    # 尝试解析 JSON
    try:
        data = json.loads(t)
        
        if isinstance(data, dict):
            if "bbox_2d" in data:
                b = safe_float_list(data["bbox_2d"])
                if b:
                    bboxes.append(b)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "bbox_2d" in item:
                    b = safe_float_list(item["bbox_2d"])
                    if b:
                        bboxes.append(b)
                elif isinstance(item, (list, tuple)):
                    b = safe_float_list(item)
                    if b:
                        bboxes.append(b)
            
            if not bboxes:
                b = safe_float_list(data)
                if b:
                    bboxes.append(b)
    except Exception:
        pass
    
    # 正则提取
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
        pixel_bbox = convert_to_pixel(bbox, img_width, img_height)
        fixed_bbox = fix_and_clip(pixel_bbox, img_width, img_height)
        result.append(fixed_bbox)
    
    return result


# ============================================================================
# 适配器
# ============================================================================

class Qwen3VLAdapter(ModelAdapter):
    """Qwen3VL 模型适配器"""
    
    def build_prompt(self, desc_parts: List[str]) -> str:
        """构造 Qwen3VL 的 prompt"""
        description = " ".join(desc_parts).strip()
        return (
            "You are a visual grounding model. Given an image and a target description, output the target bounding box.\n"
            f"Target description: {description}\n"
            "Locate the description target, output its bbox coordinates using JSON format."
        )
    
    def parse_response(self, response: str, img_width: int, img_height: int) -> List[List[float]]:
        """解析 Qwen3VL 的输出"""
        return parse_qwen3vl_bbox(response, img_width, img_height)
    
    def create_engine(self, args):
        """创建 Qwen3VL 推理引擎"""
        if args.mode == 'local':
            return Qwen3VLLocalEngine(args.model_path)
        else:
            return Qwen3VLAPIEngine(
                api_key=args.api_key,
                api_base_url=args.api_base_url,
                model_name=args.api_model_name,
                temperature=args.api_temperature,
                max_tokens=args.api_max_tokens,
                retries=args.api_retries,
            )
    
    def get_default_model_path(self) -> str:
        return _pick_existing([
            os.environ.get('QWEN3VL_MODEL_PATH'),
            '/root/user-data/MODEL_WEIGHTS_PUBLIC/MLLM_weights/Qwen3-VL-32B-Instruct',
            '/root/user-data/MODEL_WEIGHTS_PUBLIC/Qwen3-VL-32B-Instruct',
            '/home/member/data1/MODEL_WEIGHTS_PUBLIC/Qwen3-VL-32B-Instruct/',
        ])
    
    def get_default_api_model_name(self) -> str:
        return "qwen3-vl-32b-instruct"
    
    def get_default_api_base_url(self) -> str:
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"

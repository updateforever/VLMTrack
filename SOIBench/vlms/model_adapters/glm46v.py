# -*- coding: utf-8 -*-
"""
SOIBench/vlms/model_adapters/glm46v.py
GLM-4.6V 模型完整实现
包含推理引擎、bbox 解析和适配器
"""

import os
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


def _default_glm_path():
    return _pick_existing([
        os.environ.get('GLM46V_MODEL_PATH'),
        '/root/user-data/MODEL_WEIGHTS_PUBLIC/GLM-4.6V-Flash',
        '/home/member/data1/MODEL_WEIGHTS_PUBLIC/GLM-4.6V-Flash/',
    ])

class GLM46VLocalEngine:
    """GLM-4.6V 本地推理引擎"""
    
    def __init__(self, model_path: str, device_map: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print(f"🚀 加载 GLM-4.6V 本地模型: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=device_map,
            trust_remote_code=True
        ).eval()
        
        print("✅ GLM-4.6V 模型加载完成")
    
    def chat(self, image_path: str, prompt: str, max_new_tokens: int = 512) -> str:
        """单张图推理"""
        import torch
        from PIL import Image
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 构造输入
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "image": image, "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        )
        
        inputs = inputs.to(self.model.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.1,
            )
        
        # 解码
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=False
        )
        
        return response


class GLM46VAPIEngine:
    """GLM-4.6V API 推理引擎"""
    
    def __init__(
        self,
        api_key: str = None,
        api_base_url: str = "https://api.siliconflow.cn/v1",
        model_name: str = "zai-org/GLM-4.6V",
        temperature: float = 0.1,
        max_tokens: int = 512,
        retries: int = 3,
    ):
        from openai import OpenAI
        
        if api_key is None:
            api_key = os.getenv("SILICONFLOW_API_KEY") or os.getenv("ZHIPUAI_API_KEY")
            if not api_key:
                raise ValueError("请设置 SILICONFLOW_API_KEY 或 ZHIPUAI_API_KEY 环境变量")
        
        self.client = OpenAI(api_key=api_key, base_url=api_base_url)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        
        print(f"🚀 初始化 GLM-4.6V API 引擎")
        print(f"   模型: {model_name}")
        print(f"   Base URL: {api_base_url}")
    
    def chat(self, image_path: str, prompt: str, max_new_tokens: int = None) -> str:
        """单张图推理"""
        import base64
        import time
        
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        for attempt in range(self.retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_new_tokens or self.max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < self.retries - 1:
                    wait_time = 2 ** attempt
                    print(f"⚠️  API 调用失败 (尝试 {attempt + 1}/{self.retries}): {e}")
                    print(f"   等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ API 调用失败: {e}")
                    raise


# ============================================================================
# Bbox 解析
# ============================================================================

def parse_glm46v_bbox(response: str, image_width: int, image_height: int) -> List[List[float]]:
    """
    解析 GLM-4.6V 的 bbox 输出
    
    输出格式:
    - 使用 <|begin_of_box|> 和 <|end_of_box|> 标记
    - 坐标格式: [x1, y1, x2, y2]
    - 坐标值是归一化后乘以 1000 的整数
    """
    bboxes = []
    
    # 提取 <|begin_of_box|> 和 <|end_of_box|> 之间的内容
    box_pattern = r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>'
    matches = re.findall(box_pattern, response, re.DOTALL)
    
    for match in matches:
        coord_pattern = r'[\[\(<]?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*[\]\)>]?'
        coords = re.findall(coord_pattern, match)
        
        for coord in coords:
            try:
                # 归一化坐标 (0-1000) -> 像素坐标
                x1_norm, y1_norm, x2_norm, y2_norm = map(int, coord)
                
                x1 = (x1_norm / 1000.0) * image_width
                y1 = (y1_norm / 1000.0) * image_height
                x2 = (x2_norm / 1000.0) * image_width
                y2 = (y2_norm / 1000.0) * image_height
                
                # 确保坐标在图像范围内
                x1 = max(0, min(x1, image_width))
                y1 = max(0, min(y1, image_height))
                x2 = max(0, min(x2, image_width))
                y2 = max(0, min(y2, image_height))
                
                bboxes.append([x1, y1, x2, y2])
            except (ValueError, IndexError) as e:
                print(f"⚠️  解析坐标失败: {coord}, 错误: {e}")
                continue
    
    # 如果没有找到特殊标记，尝试直接提取数字
    if not bboxes:
        fallback_pattern = r'\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?'
        coords = re.findall(fallback_pattern, response)
        
        for coord in coords:
            try:
                x1_norm, y1_norm, x2_norm, y2_norm = map(int, coord)
                
                if all(0 <= c <= 1000 for c in [x1_norm, y1_norm, x2_norm, y2_norm]):
                    x1 = (x1_norm / 1000.0) * image_width
                    y1 = (y1_norm / 1000.0) * image_height
                    x2 = (x2_norm / 1000.0) * image_width
                    y2 = (y2_norm / 1000.0) * image_height
                    
                    x1 = max(0, min(x1, image_width))
                    y1 = max(0, min(y1, image_height))
                    x2 = max(0, min(x2, image_width))
                    y2 = max(0, min(y2, image_height))
                    
                    bboxes.append([x1, y1, x2, y2])
            except (ValueError, IndexError):
                continue
    
    return bboxes


# ============================================================================
# 适配器
# ============================================================================

class GLM46VAdapter(ModelAdapter):
    """GLM-4.6V 模型适配器"""
    
    def build_prompt(self, desc_parts: List[str]) -> str:
        """构造 GLM-4.6V 的 prompt"""
        description = " ".join(desc_parts).strip()
        return f"Please pinpoint the bounding box in the image as per the given description: {description}"
    
    def parse_response(self, response: str, img_width: int, img_height: int) -> List[List[float]]:
        """解析 GLM-4.6V 的输出"""
        return parse_glm46v_bbox(response, img_width, img_height)
    
    def create_engine(self, args):
        """创建 GLM-4.6V 推理引擎"""
        if args.mode == 'local':
            return GLM46VLocalEngine(args.model_path)
        else:
            return GLM46VAPIEngine(
                api_key=args.api_key,
                api_base_url=args.api_base_url,
                model_name=args.api_model_name,
                temperature=args.api_temperature,
                max_tokens=args.api_max_tokens,
                retries=args.api_retries,
            )
    
    def get_default_model_path(self) -> str:
        return _default_glm_path()
    
    def get_default_api_model_name(self) -> str:
        return "zai-org/GLM-4.6V"
    
    def get_default_api_base_url(self) -> str:
        return "https://api.siliconflow.cn/v1"

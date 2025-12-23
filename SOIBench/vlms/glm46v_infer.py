# -*- coding: utf-8 -*-
"""
SOIBench/vlms/glm46v_infer.py
GLM-4.6V 推理引擎
支持本地模型和 API 两种模式
默认使用硅基流动 API
"""

import os
import re
import json
import torch
from PIL import Image


class GLM46VLocalEngine:
    """
    GLM-4.6V 本地推理引擎
    使用 transformers 加载模型
    """
    
    def __init__(
        self,
        model_path: str = "/home/member/data1/MODEL_WEIGHTS_PUBLIC/GLM-4.6V-Flash/",
        device_map: str = "auto",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
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
        """
        单张图推理
        
        参数:
            image_path: 图像路径
            prompt: 文本提示（目标描述）
            max_new_tokens: 最大生成 token 数
            
        返回:
            模型输出文本
        """
        # 构造 grounding prompt
        full_prompt = f"Please pinpoint the bounding box in the image as per the given description: {prompt}"
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 构造输入
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "image": image, "content": full_prompt}],
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
                do_sample=False,  # 使用贪婪解码
                temperature=0.1,
            )
        
        # 解码
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=False
        )
        
        return response


class GLM46VAPIEngine:
    """
    GLM-4.6V API 推理引擎
    使用 OpenAI 兼容接口
    默认使用硅基流动 API
    """
    
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
        
        # 获取 API Key
        if api_key is None:
            # 优先使用 SILICONFLOW_API_KEY，其次是 ZHIPUAI_API_KEY
            api_key = os.getenv("SILICONFLOW_API_KEY") or os.getenv("ZHIPUAI_API_KEY")
            if not api_key:
                raise ValueError("请设置 SILICONFLOW_API_KEY 或 ZHIPUAI_API_KEY 环境变量，或传入 api_key 参数")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base_url
        )
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        
        print(f"🚀 初始化 GLM-4.6V API 引擎")
        print(f"   模型: {model_name}")
        print(f"   Base URL: {api_base_url}")
    
    def chat(self, image_path: str, prompt: str, max_new_tokens: int = None) -> str:
        """
        单张图推理
        
        参数:
            image_path: 图像路径
            prompt: 文本提示（目标描述）
            max_new_tokens: 最大生成 token 数（可选，覆盖默认值）
            
        返回:
            模型输出文本
        """
        import base64
        import time
        
        # 读取并编码图像
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # 构造 grounding prompt
        full_prompt = f"Please pinpoint the bounding box in the image as per the given description: {prompt}"
        
        # 构造消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    },
                    {
                        "type": "text",
                        "text": full_prompt
                    }
                ]
            }
        ]
        
        # 重试逻辑
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
                    print(f"❌ API 调用失败，已达最大重试次数: {e}")
                    raise


def parse_glm46v_bbox(response: str, image_width: int, image_height: int):
    """
    解析 GLM-4.6V 的 bbox 输出
    
    GLM-4.6V 输出格式:
    - 使用 <|begin_of_box|> 和 <|end_of_box|> 标记
    - 坐标格式: [x1, y1, x2, y2] 或 [[x1, y1, x2, y2]]
    - 坐标值是归一化后乘以 1000 的整数
    
    参数:
        response: 模型输出文本
        image_width: 图像宽度
        image_height: 图像高度
        
    返回:
        [[x1, y1, x2, y2], ...] 像素坐标列表
    """
    bboxes = []
    
    # 提取 <|begin_of_box|> 和 <|end_of_box|> 之间的内容
    box_pattern = r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>'
    matches = re.findall(box_pattern, response, re.DOTALL)
    
    for match in matches:
        # 提取数字坐标
        # 支持多种括号格式: [], [[]], (), <>, 等
        coord_pattern = r'[\[\(<]?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*[\]\)>]?'
        coords = re.findall(coord_pattern, match)
        
        for coord in coords:
            try:
                # 归一化坐标 (0-1000) -> 像素坐标
                x1_norm, y1_norm, x2_norm, y2_norm = map(int, coord)
                
                # 转换为像素坐标
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
        # 尝试提取任何形式的四元组数字
        fallback_pattern = r'\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?'
        coords = re.findall(fallback_pattern, response)
        
        for coord in coords:
            try:
                x1_norm, y1_norm, x2_norm, y2_norm = map(int, coord)
                
                # 检查是否是归一化坐标 (0-1000)
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


# 测试代码
if __name__ == "__main__":
    # 测试本地模型
    # engine = GLM46VLocalEngine()
    
    # 测试硅基流动 API
    engine = GLM46VAPIEngine()
    
    # 测试推理
    image_path = "test.jpg"
    prompt = "A red car"
    
    response = engine.chat(image_path, prompt)
    print(f"模型输出: {response}")
    
    # 解析 bbox
    img = Image.open(image_path)
    bboxes = parse_glm46v_bbox(response, img.width, img.height)
    print(f"解析的 bbox: {bboxes}")
# -*- coding: utf-8 -*-
"""
SOIBench/vlms/deepseekvl_infer.py
DeepSeek-VL2 推理引擎
支持本地模型和 API 两种模式
默认使用硅基流动 API
"""

import os
import re
import json
import torch
from PIL import Image


class DeepSeekVLLocalEngine:
    """
    DeepSeek-VL2 本地推理引擎
    使用 transformers 加载模型
    """
    
    def __init__(
        self,
        model_path: str = "/home/member/data1/MODEL_WEIGHTS_PUBLIC/deepseek-vl2-small/",
        device_map: str = "auto",
    ):
        from transformers import AutoModelForCausalLM
        from deepseek_vl.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
        from deepseek_vl.utils.io import load_pil_images
        
        print(f"🚀 加载 DeepSeek-VL2 本地模型: {model_path}")
        
        self.vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        
        self.vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()
        
        self.load_pil_images = load_pil_images
        
        print("✅ DeepSeek-VL2 模型加载完成")
    
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
        # 构造 grounding prompt (使用 <|ref|> 标记)
        full_prompt = f"<image>\n<|ref|>{prompt}<|/ref|>."
        
        # 构造对话
        conversation = [
            {
                "role": "<|User|>",
                "content": full_prompt,
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        # 加载图像
        pil_images = self.load_pil_images(conversation)
        
        # 准备输入
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(self.vl_gpt.device)
        
        # 获取图像嵌入
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        
        # 生成
        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True
        )
        
        # 解码
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        
        return answer


class DeepSeekVLAPIEngine:
    """
    DeepSeek-VL2 API 推理引擎
    使用 OpenAI 兼容接口
    默认使用硅基流动 API
    """
    
    def __init__(
        self,
        api_key: str = None,
        api_base_url: str = "https://api.siliconflow.cn/v1",
        model_name: str = "deepseek-ai/deepseek-vl2",
        temperature: float = 0.1,
        max_tokens: int = 512,
        retries: int = 3,
    ):
        from openai import OpenAI
        
        # 获取 API Key
        if api_key is None:
            # 优先使用 SILICONFLOW_API_KEY，其次是 DEEPSEEK_API_KEY
            api_key = os.getenv("SILICONFLOW_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("请设置 SILICONFLOW_API_KEY 或 DEEPSEEK_API_KEY 环境变量，或传入 api_key 参数")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base_url
        )
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        
        print(f"🚀 初始化 DeepSeek-VL2 API 引擎")
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
        
        # 构造 grounding prompt (使用 <|ref|> 标记)
        full_prompt = f"<|ref|>{prompt}<|/ref|>."
        
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


def parse_deepseekvl_bbox(response: str, image_width: int, image_height: int):
    """
    解析 DeepSeek-VL2 的 bbox 输出
    
    DeepSeek-VL2 输出格式:
    - 使用 <box> 和 </box> 标记
    - 坐标格式: [[x1, y1, x2, y2]]
    - 坐标值是归一化坐标 (0-1000)
    
    参数:
        response: 模型输出文本
        image_width: 图像宽度
        image_height: 图像高度
        
    返回:
        [[x1, y1, x2, y2], ...] 像素坐标列表
    """
    bboxes = []
    
    # 提取 <box> 和 </box> 之间的内容
    box_pattern = r'<box>(.*?)</box>'
    matches = re.findall(box_pattern, response, re.DOTALL)
    
    for match in matches:
        # 提取数字坐标
        # 支持格式: [[x1, y1, x2, y2]] 或 [x1, y1, x2, y2]
        coord_pattern = r'\[+\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]+'
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
    
    # 如果没有找到 <box> 标记，尝试直接提取数字
    if not bboxes:
        # 尝试提取任何形式的四元组数字
        fallback_pattern = r'\[+\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]+'
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
    # 测试硅基流动 API
    engine = DeepSeekVLAPIEngine()
    
    # 测试推理
    image_path = "test.jpg"
    prompt = "a red car"
    
    response = engine.chat(image_path, prompt)
    print(f"模型输出: {response}")
    
    # 解析 bbox
    img = Image.open(image_path)
    bboxes = parse_deepseekvl_bbox(response, img.width, img.height)
    print(f"解析的 bbox: {bboxes}")

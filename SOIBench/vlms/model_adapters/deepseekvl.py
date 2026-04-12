# -*- coding: utf-8 -*-
"""
SOIBench/vlms/model_adapters/deepseekvl.py
DeepSeek-VL2 模型完整实现
包含推理引擎、bbox 解析和适配器
"""

import os
import re
import time
import base64
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


def _default_deepseek_path():
    return _pick_existing([
        os.environ.get('DEEPSEEKVL_MODEL_PATH'),
        '/root/user-data/MODEL_WEIGHTS_PUBLIC/deepseek-vl2-small',
        '/home/member/data1/MODEL_WEIGHTS_PUBLIC/deepseek-vl2-small/',
    ])

class DeepSeekVLLocalEngine:
    """DeepSeek-VL2 本地推理引擎"""
    
    def __init__(
        self,
        model_path: str = None,
        device_map: str = "auto",
    ):
        import torch
        from transformers import AutoModelForCausalLM
        from deepseek_vl.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
        from deepseek_vl.utils.io import load_pil_images
        
        model_path = model_path or _default_deepseek_path()
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
        """单张图推理"""
        # 构造对话
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{prompt}",
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
    """DeepSeek-VL2 API 推理引擎"""
    
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
        
        if api_key is None:
            api_key = os.getenv("SILICONFLOW_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("请设置 SILICONFLOW_API_KEY 或 DEEPSEEK_API_KEY 环境变量")
        
        self.client = OpenAI(api_key=api_key, base_url=api_base_url)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        
        print(f"🚀 初始化 DeepSeek-VL2 API 引擎")
        print(f"   模型: {model_name}")
        print(f"   Base URL: {api_base_url}")
    
    def chat(self, image_path: str, prompt: str, max_new_tokens: int = None) -> str:
        """单张图推理"""
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

def parse_deepseekvl_bbox(response: str, image_width: int, image_height: int) -> List[List[float]]:
    """
    解析 DeepSeek-VL2 的 bbox 输出
    
    输出格式:
    - 使用 <box> 和 </box> 标记
    - 坐标格式: [[x1, y1, x2, y2]]
    - 坐标值是归一化坐标 (0-1000)
    """
    bboxes = []
    
    # 提取 <box> 和 </box> 之间的内容
    box_pattern = r'<box>(.*?)</box>'
    matches = re.findall(box_pattern, response, re.DOTALL)
    
    for match in matches:
        coord_pattern = r'\[+\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]+'
        coords = re.findall(coord_pattern, match)
        
        for coord in coords:
            try:
                x1_norm, y1_norm, x2_norm, y2_norm = map(int, coord)
                
                x1 = (x1_norm / 1000.0) * image_width
                y1 = (y1_norm / 1000.0) * image_height
                x2 = (x2_norm / 1000.0) * image_width
                y2 = (y2_norm / 1000.0) * image_height
                
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
        fallback_pattern = r'\[+\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]+'
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

class DeepSeekVLAdapter(ModelAdapter):
    """DeepSeek-VL2 模型适配器"""
    
    def build_prompt(self, desc_parts: List[str]) -> str:
        """构造 DeepSeek-VL2 的 prompt"""
        description = " ".join(desc_parts).strip()
        # DeepSeek-VL2 需要 <|ref|> 标记进行 grounding
        return f"<|ref|>{description}<|/ref|>."
    
    def parse_response(self, response: str, img_width: int, img_height: int) -> List[List[float]]:
        """解析 DeepSeek-VL2 的输出"""
        return parse_deepseekvl_bbox(response, img_width, img_height)
    
    def create_engine(self, args):
        """创建 DeepSeek-VL2 推理引擎"""
        if args.mode == 'local':
            return DeepSeekVLLocalEngine(args.model_path)
        else:
            return DeepSeekVLAPIEngine(
                api_key=args.api_key,
                api_base_url=args.api_base_url,
                model_name=args.api_model_name,
                temperature=args.api_temperature,
                max_tokens=args.api_max_tokens,
                retries=args.api_retries,
            )
    
    def get_default_model_path(self) -> str:
        return _default_deepseek_path()
    
    def get_default_api_model_name(self) -> str:
        return "deepseek-ai/deepseek-vl2"
    
    def get_default_api_base_url(self) -> str:
        return "https://api.siliconflow.cn/v1"

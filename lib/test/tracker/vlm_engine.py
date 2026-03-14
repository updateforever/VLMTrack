"""
VLM推理引擎基类

提供统一的VLM推理接口，支持：
- 本地推理 (Local)
- API推理 (API)
"""
import torch
import os
from PIL import Image
import numpy as np
from typing import List

from lib.test.tracker.vlm_utils import numpy_to_base64, call_vlm_api


class VLMEngine:
    """
    VLM推理引擎
    
    支持两种模式:
    - mode='local': 本地加载模型推理
    - mode='api': 调用API推理
    """
    
    def __init__(self, params):
        """
        初始化VLM引擎
        
        Args:
            params: 参数对象，需包含:
                - mode: 'local' 或 'api'
                - model_name/model_path: 本地模式使用
                - api_model/api_base_url/api_key: API模式使用
        """
        self.params = params
        self.mode = getattr(params, 'mode', 'local')
        
        if self.mode == 'local':
            self._load_local_model()
        else:
            self._setup_api()
    
    def _load_local_model(self):
        """加载本地模型"""
        from transformers import AutoProcessor

        model_path = getattr(self.params, 'model_path', None)
        model_name = getattr(self.params, 'model_name', 'Qwen/Qwen3-VL-4B-Instruct')
        actual_path = model_path or model_name

        # 根据模型名选择模型类
        if 'qwen3' in actual_path.lower():
            from transformers import Qwen3VLForConditionalGeneration
            model_class = Qwen3VLForConditionalGeneration
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model_class = Qwen2_5_VLForConditionalGeneration

        # 加载模型（优先 flash_attention_2，失败则降级）
        try:
            self.model = model_class.from_pretrained(
                actual_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto"
            )
        except Exception:
            self.model = model_class.from_pretrained(
                actual_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(actual_path)
    
    def _setup_api(self):
        """配置API，并缓存客户端实例"""
        from openai import OpenAI
        self.api_model = getattr(self.params, 'api_model', 'qwen3-vl-235b-a22b-instruct')
        self.api_base_url = getattr(self.params, 'api_base_url',
                                    'https://dashscope.aliyuncs.com/compatible-mode/v1')
        self.api_key = getattr(self.params, 'api_key', os.environ.get('DASHSCOPE_API_KEY', ''))
        # 缓存客户端，避免每帧重复创建
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base_url)
    
    def infer(self, images: List[np.ndarray], prompt: str) -> str:
        """
        运行VLM推理
        
        Args:
            images: RGB numpy图像列表
            prompt: 文本prompt
        
        Returns:
            模型输出文本
        """
        if self.mode == 'api':
            return self._infer_api(images, prompt)
        else:
            return self._infer_local(images, prompt)
    
    def _infer_api(self, images: List[np.ndarray], prompt: str) -> str:
        """API模式推理（复用缓存的client）"""
        images_b64 = [numpy_to_base64(img) for img in images]
        return call_vlm_api(
            images_b64=images_b64,
            prompt=prompt,
            model_name=self.api_model,
            client=self.client,
        )
    
    def _infer_local(self, images: List[np.ndarray], prompt: str) -> str:
        """本地模式推理"""
        # 转换为PIL Image
        images_pil = [Image.fromarray(img) for img in images]
        
        # 构建消息
        content = []
        for img in images_pil:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        
        # 应用chat模板
        inputs = self.processor.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True,
            return_dict=True, 
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        
        # 生成（max_new_tokens可通过params配置，默认512）
        max_new_tokens = getattr(self.params, 'max_new_tokens', 512)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        
        # 解码
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text

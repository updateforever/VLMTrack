# -*- coding: utf-8 -*-
"""
SOIBench/vlms/qwen3vl_infer.py
Qwen3 VL 推理引擎封装
目标：
1）把模型加载与推理封装成可复用模块
2）严格使用 Qwen VL 官方 messages 形式
3）修正 apply_chat_template 的用法，避免 tokenize=True 造成的输入不匹配
"""

import os
import time
import base64
from typing import List, Dict, Any

import torch


#############
import torch

def _check_tensor(name, x):
    if x is None:
        print(f"[CHK] {name}: None")
        return
    if not torch.is_tensor(x):
        print(f"[CHK] {name}: type={type(x)}")
        return
    info = {
        "dtype": str(x.dtype),
        "device": str(x.device),
        "shape": tuple(x.shape),
    }
    msg = f"[CHK] {name}: {info}"
    if x.numel() > 0 and x.dtype in (torch.int32, torch.int64, torch.long):
        msg += f", min={int(x.min())}, max={int(x.max())}"
    if x.numel() > 0 and x.is_floating_point():
        finite = torch.isfinite(x)
        msg += f", finite={bool(finite.all())}"
        if not finite.all():
            bad = (~finite).nonzero(as_tuple=False)[:5].tolist()
            msg += f", bad_idx_examples={bad}"
    print(msg)

def _check_input_ids_range(input_ids, vocab_size):
    if input_ids is None:
        raise RuntimeError("input_ids is None")
    if input_ids.dtype not in (torch.int32, torch.int64, torch.long):
        raise RuntimeError(f"input_ids dtype must be int64, got {input_ids.dtype}")
    mn = int(input_ids.min())
    mx = int(input_ids.max())
    if mn < 0 or mx >= vocab_size:
        # 找出越界位置的少量样例，便于定位是哪段 prompt 或哪类 token 造成的
        bad = ((input_ids < 0) | (input_ids >= vocab_size)).nonzero(as_tuple=False)
        bad = bad[:20].tolist()
        raise RuntimeError(
            f"input_ids out of range: min={mn}, max={mx}, vocab_size={vocab_size}, bad_pos_examples={bad}"
        )



#############

class Qwen3VLLocalEngine:
    """
    本地 Qwen3 VL 推理引擎
    使用 transformers 的 AutoModelForImageTextToText 与 AutoProcessor 加载
    """

    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        device_map: str = "auto",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
    ):
        # from transformers import AutoModelForImageTextToText, AutoProcessor
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        print(f"🚀 加载本地模型: {model_path}")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            # attn_implementation=attn_implementation,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
        )


    def chat(self, image_path: str, prompt: str, max_new_tokens: int = 256) -> str:
        """
        单张图推理，返回模型输出文本
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": image_path
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "image",
        #                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        #             },
        #             {"type": "text", "text": "Describe this image."},
        #         ],
        #     }
        # ]

        # Preparation for inference
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # 去掉 prompt 对应的 token，仅保留新生成部分
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0] or ""


def qwen3vl_api_chat(
    image_path: str,
    prompt: str,
    model_name: str,
    base_url: str,
    api_key: str = "DASHSCOPE_API_KEY",
    temperature: float = 0.1,
    max_tokens: int = 256,
    retries: int = 3,
    retry_sleep: float = 1.0,
) -> str:
    """
    OpenAI 兼容 API 推理封装
    """
    from openai import OpenAI

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    client = OpenAI(api_key=api_key, base_url=base_url)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    last_err = None
    for _ in range(max(1, retries)):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            time.sleep(retry_sleep)

    raise RuntimeError(f"API 调用失败: {last_err}")

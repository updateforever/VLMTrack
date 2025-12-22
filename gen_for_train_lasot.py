#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen-VL SOI (Similar Object Identification) 统一推理脚本
支持后端: 
1. Local (HuggingFace transformers with Batch Inference)
2. vLLM (Async API with High Concurrency)
"""

import argparse
import json
import logging
import math
import os
import re
import io
import asyncio
import aiohttp
import base64
import time
from typing import List, Dict, Optional, Set, Any, Union
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm 

# 尝试导入数据集工具
try:
    from lib.test.evaluation import get_dataset
except ImportError:
    logging.error("Error: 'lib.test.evaluation' not found. Check PYTHONPATH.")
    exit(1)

# ===========================
#   1. Prompt (双语 SOI)
# ===========================
PROMPTS = {
    "en": """
You are a visual tracking expert. 
The **Target Object** is clearly marked by a **GREEN Bounding Box**.
There are visually similar distractor objects in the scene (SOI Challenge).

# Cognitive Guidance
Your task is to generate a structured description strictly following two principles:
1. **Concretization**: Provide vivid, visible details inside the green box.
2. **Saliency Guiding**: Emphasize features that distinguish the target from distractors.

# Output Requirements
Return a JSON object with 4 levels. 
- L1-L3 should logically form a coherent sentence description.
- L4 MUST focus on **Discrimination** (Spatial relation or Action difference vs Distractors).

JSON Format:
{
    "level1": "<Location Feature (Start with preposition, e.g., 'In the center of the road,')>",
    "level2": "<Appearance Description (e.g., 'a white car', mention carrier if any)>",
    "level3": "<Dynamic State (Verb phrase, e.g., 'is turning left.')>",
    "level4": "<Distractor Differentiation (e.g., 'It is located to the left of another white car.')>"
}
Output ONLY the JSON object.
""",
    "cn": """
你正在观察一张包含待跟踪目标的图像，目标物体已被**绿色框**清晰标注。
图中可能存在多个与目标外观相似的干扰物 (SOI挑战)。

# 任务说明
你的任务是为被跟踪的目标生成一份简洁、结构化的多层次语义描述，严格遵循以下两个认知语言学原则：
1. **具象化**（提供生动、具体、可直观想象的细节）
2. **显著性引导**（突出能快速区分目标与干扰物的显著特征）

# 输出要求
请返回一个包含四个层级的 JSON 对象。前三个层级（L1-L3）拼接起来应是一句通顺的中文描述，L4 单独用于区分干扰物。

JSON 格式如下：
{
    "level1": "<位置特征 (以介词开头，如'在马路中央，'，描述背景相对位置)>",
    "level2": "<外观描述 (如'一辆白色的SUV'，若有载体需描述，如'被拿着的')>",
    "level3": "<动作状态 (完整的动词短语，如'正在向左转弯。')>",
    "level4": "<干扰物区分 (核心层级：请明确指出目标与相似物体的区别，例如'它位于另一辆相似白车的左侧。' 或 '它是唯一站立的那只。')>"
}
仅输出 JSON 对象，不要包含任何解释。
"""
}

# ===========================
#   2. 通用工具函数
# ===========================

def get_category_from_seq(seq_name: str) -> str:
    if '-' in seq_name:
        return seq_name.rsplit('-', 1)[0]
    return seq_name

def read_selection_indices(jsonl_root: str, seq_name: str) -> Optional[Set[int]]:
    """智能路径查找：支持 root/category/seq.jsonl 或 root/seq.jsonl"""
    root = Path(jsonl_root)
    cat = get_category_from_seq(seq_name)
    
    candidates = [
        root / cat / f"{seq_name}.jsonl",
        root / f"{seq_name}.jsonl"
    ]
    
    target = next((p for p in candidates if p.exists()), None)
    if not target: return None

    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list): return set(data)
            if isinstance(data, dict) and "changes" in data: return set(data["changes"])
            return None
    except:
        return None

def draw_bbox(image_path: str, bbox: List[float]) -> Image.Image:
    """基础绘图函数，返回 PIL Image"""
    img = Image.open(image_path).convert("RGB")
    x, y, w, h = bbox
    w_img, h_img = img.size
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    
    draw = ImageDraw.Draw(img)
    draw.rectangle([x, y, x + w, y + h], outline="lime", width=4)
    return img

def clean_json(text: str) -> Optional[Dict]:
    """鲁棒的 JSON 清洗"""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    json_str = match.group(0) if match else text
    try:
        return json.loads(json_str)
    except:
        try:
            return json.loads(json_str.replace("'", '"'))
        except:
            return None

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# ===========================
#   3. 引擎定义
# ===========================

# --- 引擎 A: Local Transformers (支持 Batch) ---
class LocalBatchEngine:
    def __init__(self, model_path, batch_size=4):
        try:
            import torch
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        except ImportError:
            logging.error("Local mode requires: pip install torch transformers qwen-vl-utils")
            exit(1)

        logging.info(f"[Local] Loading model: {model_path}")
        self.batch_size = batch_size
        
        # 自动适配 H800 bf16 和 FlashAttention
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto",
            # attn_implementation="flash_attention_2" # 如果报错不支持，可改为 "sdpa" 或删掉此行
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model.eval()
        logging.info(f"[Local] Model loaded. Batch Size: {batch_size}")

    def infer_batch(self, tasks: List[Dict], prompt_text: str) -> List[str]:
        images = [t['image'] for t in tasks]
        
        # 构造 Batch Messages
        messages_batch = []
        for img in images:
            messages_batch.append([
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt_text}
                ]}
            ])
        
        # 预处理
        text_inputs = self.processor.apply_chat_template(
            messages_batch, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=text_inputs,
            images=images,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        # 推理
        import torch
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.9
            )
        
        # 解码
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_texts

# --- 引擎 B: Async vLLM (支持高并发) ---
class AsyncVLLMEngine:
    def __init__(self, url, concurrency=50):
        self.url = url
        self.sem = asyncio.Semaphore(concurrency)

    async def infer(self, session, b64_img, prompt):
        payload = {
            "model": "qwen-vl",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            "temperature": 0.1,
            "max_tokens": 512
        }
        async with self.sem:
            try:
                async with session.post(self.url, json=payload, headers={"Authorization": "Bearer EMPTY"}) as resp:
                    if resp.status == 200:
                        res = await resp.json()
                        return res["choices"][0]["message"]["content"]
                    return None
            except:
                return None

# ===========================
#   4. 执行逻辑 (Local vs vLLM)
# ===========================

def run_local(args, dataset):
    """Local 模式主循环 (支持断点续传 + 实时保存)"""
    logging.info(">>> Running in LOCAL mode")
    engine = LocalBatchEngine(args.model_path, args.batch_size)
    
    for seq in dataset:
        seq_name = seq.name
        output_path = Path(args.output_dir) / f"{seq_name}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # --- [新增] 断点续传逻辑 ---
        processed_indices = set()
        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        processed_indices.add(data["frame_index"])
                    except: pass
            logging.info(f"[{seq_name}] Resuming... Found {len(processed_indices)} processed frames.")
        
        indices = read_selection_indices(args.jsonl_root, seq_name)
        if not indices: continue

        # 准备数据 (过滤掉已处理的)
        frames = seq.frames
        gt = seq.ground_truth_rect
        task_buffer = []

        for idx in indices:
            if idx in processed_indices: continue # 跳过已处理
            if idx >= len(frames): continue
            bbox = list(map(float, gt[idx]))
            if any(math.isnan(x) for x in bbox): continue
            
            try:
                img = draw_bbox(frames[idx], bbox)
                task_buffer.append({
                    "seq": seq_name, "idx": idx, "bbox": bbox, "path": frames[idx], "image": img
                })
            except: pass

        if not task_buffer: continue

        logging.info(f"Processing {seq_name}: {len(task_buffer)} new frames")
        
        # --- [修改] 打开文件使用 'a' 模式，且放在 Batch 循环外 ---
        with open(output_path, "a", encoding="utf-8") as f:
            # 分批处理
            for batch in tqdm(chunk_list(task_buffer, args.batch_size), total=math.ceil(len(task_buffer)/args.batch_size), desc=seq_name):
                try:
                    raw_texts = engine.infer_batch(batch, PROMPTS[args.lang])
                    
                    # 生成完一个Batch，立刻写入
                    for task, raw in zip(batch, raw_texts):
                        desc = clean_json(raw)
                        record = {
                            "video_name": task["seq"],
                            "frame_index": task["idx"],
                            "frame_path": task["path"],
                            "bbox": task["bbox"],
                            "description": desc if desc else {"raw": raw, "error": "parse"},
                            "raw_response": raw
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    
                    # 强制刷入硬盘，防止系统缓存
                    f.flush() 
                    
                except Exception as e:
                    logging.error(f"Batch Error: {e}")
                    import torch
                    torch.cuda.empty_cache()

async def run_vllm_async(args, dataset):
    """vLLM 模式主循环 (Async + Streaming Save)"""
    logging.info(">>> Running in vLLM mode")
    engine = AsyncVLLMEngine(args.vllm_url, args.concurrency)
    
    # ... (保持 encode_image 等辅助函数不变) ...
    def encode_image(path, bbox):
        try:
            img = draw_bbox(path, bbox)
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode()
        except: return None

    async def process_seq(seq, session):
        seq_name = seq.name
        output_path = Path(args.output_dir) / f"{seq_name}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # --- [新增] 断点续传 ---
        processed_indices = set()
        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        processed_indices.add(data["frame_index"])
                    except: pass
        
        indices = read_selection_indices(args.jsonl_root, seq_name)
        if not indices: return

        frames = seq.frames
        gt = seq.ground_truth_rect
        
        # 准备待处理任务
        tasks_data = []
        for idx in indices:
            if idx in processed_indices: continue # 跳过
            if idx >= len(frames): continue
            bbox = list(map(float, gt[idx]))
            if any(math.isnan(x) for x in bbox): continue
            tasks_data.append({"seq": seq_name, "idx": idx, "bbox": bbox, "path": frames[idx]})

        if not tasks_data: 
            logging.info(f"[{seq_name}] All frames processed. Skipping.")
            return

        logging.info(f"Processing {seq_name}: {len(tasks_data)} new frames")

        # 定义 Worker (保持逻辑，稍微调整返回值结构)
        async def worker(t):
            loop = asyncio.get_event_loop()
            b64 = await loop.run_in_executor(None, encode_image, t["path"], t["bbox"])
            if not b64: return None
            
            raw = await engine.infer(session, b64, PROMPTS[args.lang])
            if raw:
                desc = clean_json(raw)
                return {
                    "video_name": t["seq"], "frame_index": t["idx"],
                    "frame_path": t["path"], "bbox": t["bbox"],
                    "description": desc if desc else {"raw": raw, "error": "parse"},
                    "raw_response": raw
                }
            return None

        # --- [核心修改] 使用 as_completed 实现完成一个存一个 ---
        import asyncio
        
        # 创建所有协程任务
        pending_tasks = [worker(t) for t in tasks_data]
        
        # 使用 'a' 模式追加写入
        with open(output_path, "a", encoding="utf-8") as f:
            # tqdm 用于显示进度
            for coro in tqdm(asyncio.as_completed(pending_tasks), total=len(pending_tasks), desc=seq_name):
                result = await coro
                if result:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush() # 确保写入磁盘

    # Session 建立
    timeout = aiohttp.ClientTimeout(total=1200)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for seq in dataset:
            await process_seq(seq, session)

# ===========================
#   5. 入口
# ===========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="lasot")
    parser.add_argument("--jsonl_root", required=True)
    parser.add_argument("--output_dir", required=True)
    
    # 核心切换开关
    parser.add_argument("--backend", default="local", choices=["local", "vllm"], help="Choose inference backend")
    
    # Local 参数
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-32B-Instruct")
    parser.add_argument("--batch_size", type=int, default=2, help="For local mode")
    
    # vLLM 参数
    parser.add_argument("--vllm_url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--concurrency", type=int, default=50, help="For vLLM mode")
    
    parser.add_argument("--lang", default="cn", choices=["cn", "en"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    dataset = get_dataset(args.dataset_name)
    
    if args.backend == "local":
        run_local(args, dataset)
    else:
        asyncio.run(run_vllm_async(args, dataset))

if __name__ == "__main__":
    main()

# /seu_share/home/luxiaobo/230248984/code/tnl2k_scene_changes_clip
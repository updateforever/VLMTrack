# -*- coding: utf-8 -*-
"""
SOIBench/vlms/grounding_common.py
Grounding 推理通用函数和主流程
支持任意 VLM 模型通过适配器接入
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageColor


_ADDITIONAL_COLORS = [name for (name, _) in ImageColor.colormap.items()]


def plot_bounding_boxes(im: Image.Image, bboxes: List[List[float]], save_path: str):
    """
    在图上画 bbox
    
    参数:
        im: PIL Image 对象
        bboxes: List[List[float]]，每个为 [x1,y1,x2,y2] 像素坐标
        save_path: 保存路径
    """
    if not bboxes:
        return
    
    img = im.copy()
    draw = ImageDraw.Draw(img)
    colors = ["red", "green", "blue", "yellow", "orange", "pink", "purple"] + _ADDITIONAL_COLORS
    
    try:
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
    except Exception:
        font = ImageFont.load_default()
    
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=3)
        draw.text((x1 + 4, y1 + 4), f"Pred-{i}", fill=color, font=font)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)


def process_description_levels(output_en: Dict) -> List[str]:
    """
    处理描述文本的四个层级，添加合适的标点
    
    参数:
        output_en: 包含 level1-4 的字典
    
    返回:
        处理后的描述文本列表
    """
    desc_parts = []
    for idx, k in enumerate(["level1", "level2", "level3", "level4"], 1):
        v = (output_en.get(k, "") or "").strip()
        if v:
            # 移除末尾的标点符号
            v = v.rstrip('.,;:!?')
            
            # 转为小写
            v = v[0].lower() + v[1:] if len(v) > 0 else v
            
            # 添加标点
            if idx in [1, 2]:  # Level 1, 2: 逗号
                v = v + ','
            else:  # Level 3, 4: 句号
                v = v + '.'
            
            desc_parts.append(v)
    
    return desc_parts


def load_and_fix_paths(jsonl_path: str, dataset_name: str, image_roots: Dict[str, str]) -> List[Dict]:
    """
    读取描述 jsonl，并把 image_path 修复为绝对路径
    抽取 output-en 的 level1 到 level4 拼成描述文本
    
    参数:
        jsonl_path: jsonl 文件路径
        dataset_name: 数据集名称
        image_roots: 数据集图像根目录字典
    
    返回:
        有效样本列表，每个样本包含:
            - original_item: 原始 JSONL 行
            - image_path: 修复后的绝对路径
            - desc_parts: 处理后的描述文本列表
            - dataset_name: 数据集名称
            - frame_idx: 帧索引
    """
    image_root = image_roots.get(dataset_name)
    if not image_root:
        return []
    
    valid = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            
            # 注意：不要跳过 skip 帧！
            # skip 只是人类标注时跳过，VLM 算法需要对所有帧都进行推理
            
            # 提取描述文本并添加合适的标点
            output_en = item.get("output-en", {}) or {}
            desc_parts = process_description_levels(output_en)
            
            if not desc_parts:
                # 如果没有描述，使用默认文本
                print(f"  ⚠️  WARNING: 序列 {os.path.basename(jsonl_path)} 的帧 {item.get('frame_idx')} 缺少描述文本，使用默认 prompt")
                desc_parts = ["the target object."]
            
            # 修复图像路径
            rel = item.get("image_path", "")
            if not rel:
                continue
            if rel.startswith("/"):
                rel = rel[1:]
            
            # 尝试多种路径组合方式
            possible_paths = [
                os.path.join(image_root, rel),
            ]
            
            # LaSOT 特殊路径: 需要在中间插入年份目录
            if len(rel) > 10:
                possible_paths.append(os.path.join(image_root, rel[6:10], rel))
            
            # MGIT/TNL2K 特殊路径
            if len(rel.split('/')) > 2:
                parts = rel.split('/')
                possible_paths.append(os.path.join(image_root, parts[1], 'imgs', parts[2][1:]))
                possible_paths.append(os.path.join(image_root, parts[1], 'imgs', parts[2]))
            
            abs_path = None
            for p in possible_paths:
                if p and os.path.exists(p):
                    abs_path = p
                    break
            
            if abs_path:
                valid.append({
                    "original_item": item,
                    "image_path": abs_path,
                    "desc_parts": desc_parts,
                    "dataset_name": dataset_name,
                    "frame_idx": item.get("frame_idx", "unknown"),
                })
    
    return valid


def _count_lines(path: str) -> int:
    """
    统计文件行数，用于断点续跑
    """
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def run_grounding_inference(
    adapter,
    engine,
    jsonl_dir: str,
    dataset_name: str,
    image_roots: Dict[str, str],
    output_dir: str,
    vis_dir: str = None,
    max_new_tokens: int = 512,
):
    """
    运行 Grounding 推理的主流程
    
    参数:
        adapter: 模型适配器实例
        engine: 推理引擎实例
        jsonl_dir: JSONL 文件目录
        dataset_name: 数据集名称
        image_roots: 图像根目录字典
        output_dir: 输出目录
        vis_dir: 可视化目录（可选）
        max_new_tokens: 最大生成 token 数
    """
    # 获取所有 JSONL 文件并排序
    jsonl_files = sorted([f for f in os.listdir(jsonl_dir) if f.endswith('_descriptions.jsonl')])
    if not jsonl_files:
        print(f"⚠️  目录为空或没有 _descriptions.jsonl 文件: {jsonl_dir}")
        return
    
    print(f"\n📂 处理数据集: {dataset_name} ({len(jsonl_files)} 个序列)")
    
    for jsonl_file in tqdm(jsonl_files, desc=f"处理 {dataset_name}", dynamic_ncols=True):
        seq_name = Path(jsonl_file).stem.replace("_descriptions", "").replace("_done", "")
        save_path = os.path.join(output_dir, f"{seq_name}_pred.jsonl")
        
        # 断点续跑: 检查已处理的行数
        processed = _count_lines(save_path)
        jsonl_path = os.path.join(jsonl_dir, jsonl_file)
        samples = load_and_fix_paths(jsonl_path, dataset_name, image_roots)
        
        if processed >= len(samples):
            continue
        
        # 从断点处继续
        for s in samples[processed:]:
            img_path = s["image_path"]
            desc_parts = s["desc_parts"]
            
            # 使用适配器构造 prompt
            prompt = adapter.build_prompt(desc_parts)
            
            # 调用推理引擎
            try:
                raw_out = engine.chat(img_path, prompt, max_new_tokens=max_new_tokens)
            except Exception as e:
                print(f"  ❌ 推理失败: {e}")
                raw_out = ""
            
            # 处理空输出
            if not raw_out:
                record = s["original_item"].copy()
                record["model_response"] = raw_out
                record["parsed_bboxes"] = []
                with open(save_path, "a", encoding="utf-8") as f_out:
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
            
            # 使用适配器解析 bbox
            with Image.open(img_path) as img:
                w, h = img.size
                parsed = adapter.parse_response(raw_out, w, h)
            
            # 保存结果
            record = s["original_item"].copy()
            record["model_response"] = raw_out
            record["parsed_bboxes"] = parsed
            
            with open(save_path, "a", encoding="utf-8") as f_out:
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            # 可选: 保存可视化
            if vis_dir and parsed:
                vis_path = os.path.join(vis_dir, f"{seq_name}_{s['frame_idx']}.jpg")
                with Image.open(img_path) as img:
                    plot_bounding_boxes(img, parsed, vis_path)

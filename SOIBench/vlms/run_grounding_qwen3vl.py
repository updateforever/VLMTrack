# -*- coding: utf-8 -*-
"""
SOIBench/vlms/run_grounding_qwen3vl.py
Grounding 主流程脚本
功能：
1）读取描述 jsonl，修复图像路径
2）调用 qwen3vl_infer.py 中的本地或 API 引擎推理
3）解析输出 bbox，并统一为像素坐标
4）保存 pred.jsonl，支持断点续跑
5）可选保存可视化结果
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path

from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageColor

from qwen3vl_infer import Qwen3VLLocalEngine, qwen3vl_api_chat


_ADDITIONAL_COLORS = [name for (name, _) in ImageColor.colormap.items()]


def plot_bounding_boxes(im: Image.Image, bboxes, save_path: str):
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


def _strip_code_fence(text: str) -> str:
    """
    去掉 ```json / ``` 等代码块包裹
    """
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _safe_float_list(x):
    """
    尝试把输入转换成长度为 4 的 float list
    """
    if isinstance(x, (list, tuple)) and len(x) == 4:
        try:
            return [float(v) for v in x]
        except Exception:
            return None
    return None


def _clamp(v, lo, hi):
    """将值限制在 [lo, hi] 范围内"""
    return max(lo, min(hi, v))


def _fix_and_clip_bbox(b, w, h):
    """
    修正 bbox 坐标顺序并裁剪到图像范围内
    参数:
        b: [x1, y1, x2, y2]
        w: 图像宽度
        h: 图像高度
    返回:
        修正后的 bbox
    """
    x1, y1, x2, y2 = b
    # 确保 x1 < x2, y1 < y2
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    # 裁剪到图像范围
    x1 = _clamp(x1, 0.0, float(w - 1))
    y1 = _clamp(y1, 0.0, float(h - 1))
    x2 = _clamp(x2, 0.0, float(w - 1))
    y2 = _clamp(y2, 0.0, float(h - 1))

    # 确保 bbox 至少有 1 像素宽高
    if abs(x2 - x1) < 1.0:
        x2 = _clamp(x1 + 1.0, 0.0, float(w - 1))
    if abs(y2 - y1) < 1.0:
        y2 = _clamp(y1 + 1.0, 0.0, float(h - 1))

    return [x1, y1, x2, y2]


def _convert_to_pixel_bbox(b, w, h):
    """
    支持三类坐标体系并统一成像素坐标
    1）0 到 1 归一化坐标
    2）0 到 1000 归一化坐标
    3）像素坐标
    参数:
        b: [x1, y1, x2, y2]
        w: 图像宽度
        h: 图像高度
    返回:
        像素坐标 [x1, y1, x2, y2]
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


def extract_bboxes_from_model_output(text: str, img_width: int, img_height: int):
    """
    从模型输出中解析 bbox
    支持多种格式:
    1）JSON: {"bbox_2d":[...]} 或 [{"bbox_2d":[...]}]
    2）JSON: [x1,y1,x2,y2]
    3）文本中包含多个 [x1,y1,x2,y2]
    4）兼容 0 到 1，0 到 1000，像素坐标
    
    参数:
        text: 模型输出文本
        img_width: 图像宽度
        img_height: 图像高度
    返回:
        List[[x1,y1,x2,y2]]，像素坐标
    """
    raw = text or ""
    t = _strip_code_fence(raw)
    if not t:
        return []

    bboxes = []

    # 尝试解析 JSON
    try:
        data = json.loads(t)

        if isinstance(data, dict):
            # 单个 bbox: {"bbox_2d": [x1,y1,x2,y2]}
            if "bbox_2d" in data:
                b = _safe_float_list(data["bbox_2d"])
                if b:
                    b = _convert_to_pixel_bbox(b, img_width, img_height)
                    bboxes.append(_fix_and_clip_bbox(b, img_width, img_height))

            # 多个 bbox: {"bboxes": [{...}, {...}]}
            if "bboxes" in data and isinstance(data["bboxes"], list):
                for it in data["bboxes"]:
                    if isinstance(it, dict) and "bbox_2d" in it:
                        b = _safe_float_list(it["bbox_2d"])
                    else:
                        b = _safe_float_list(it)
                    if b:
                        b = _convert_to_pixel_bbox(b, img_width, img_height)
                        bboxes.append(_fix_and_clip_bbox(b, img_width, img_height))

        elif isinstance(data, list):
            # 直接是一个 bbox: [x1,y1,x2,y2]
            if len(data) == 4 and all(isinstance(x, (int, float)) for x in data):
                b = [float(x) for x in data]
                b = _convert_to_pixel_bbox(b, img_width, img_height)
                bboxes.append(_fix_and_clip_bbox(b, img_width, img_height))
            else:
                # 多个 bbox: [[x1,y1,x2,y2], ...] 或 [{"bbox_2d": ...}, ...]
                for it in data:
                    if isinstance(it, dict) and "bbox_2d" in it:
                        b = _safe_float_list(it["bbox_2d"])
                    else:
                        b = _safe_float_list(it)
                    if b:
                        b = _convert_to_pixel_bbox(b, img_width, img_height)
                        bboxes.append(_fix_and_clip_bbox(b, img_width, img_height))

        if bboxes:
            return bboxes

    except Exception:
        pass

    # JSON 解析失败，尝试正则匹配 [x1,y1,x2,y2]
    matches = re.findall(r"\[([^\[\]]+)\]", t)
    for m in matches:
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m)
        if len(nums) == 4:
            try:
                b = [float(x) for x in nums]
                b = _convert_to_pixel_bbox(b, img_width, img_height)
                bboxes.append(_fix_and_clip_bbox(b, img_width, img_height))
            except Exception:
                continue

    return bboxes


def load_and_fix_paths(jsonl_path: str, dataset_name: str, image_roots: dict):
    """
    读取描述 jsonl，并把 image_path 修复为绝对路径
    抽取 output-en 的 level1 到 level4 拼成描述文本
    
    参数:
        jsonl_path: jsonl 文件路径
        dataset_name: 数据集名称
        image_roots: 数据集图像根目录字典
    返回:
        有效样本列表
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
            
            # 提取描述文本
            output_en = item.get("output-en", {}) or {}
            desc_parts = []
            for k in ["level1", "level2", "level3", "level4"]:
                v = (output_en.get(k, "") or "").strip()
                if v:
                    desc_parts.append(v)

            full_desc = " ".join(desc_parts).strip()
            if not full_desc:
                # 如果没有描述，使用默认文本
                print(f"  ⚠️  WARNING: 序列 {os.path.basename(jsonl_path)} 的帧 {item.get('frame_idx')} 缺少描述文本，使用默认 prompt")
                full_desc = "the target object"

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
                    "text_prompt": full_desc,
                    "dataset_name": dataset_name,
                    "frame_idx": item.get("frame_idx", "unknown"),
                })

    return valid


def build_prompt(description: str) -> str:
    """
    构建 Grounding prompt
    """
    return (
        "You are a visual grounding model. Given an image and a target description, output the target bounding box.\n"
        f"Target description: {description}\n"
        "Locate the description target, output its bbox coordinates using JSON format."
    )


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


def main():
    parser = argparse.ArgumentParser(description="SOIBench Grounding 评测脚本")

    # 推理模式
    parser.add_argument("--mode", type=str, default="local", choices=["local", "api"],
                        help="推理模式: local(本地) 或 api(API调用)")
    parser.add_argument("--model_path", type=str, default="",
                        help="本地模型路径 (mode=local 时必需)")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="最大生成 token 数")

    # API 配置
    parser.add_argument("--api_model_name", type=str, default="qwen-vl-max",
                        help="API 模型名称")
    parser.add_argument("--api_base_url", type=str, 
                        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        help="API base URL")
    parser.add_argument("--api_key_env", type=str, default="sk-61547e720ce8407aa44f4511051903b0",
                        help="API key")
    parser.add_argument("--api_temperature", type=float, default=0.1,
                        help="API 温度参数")
    parser.add_argument("--api_max_tokens", type=int, default=256,
                        help="API 最大 token 数")
    parser.add_argument("--api_retries", type=int, default=3,
                        help="API 重试次数")

    # 实验配置
    parser.add_argument("--exp_tag", type=str, default="run",
                        help="实验标签")
    parser.add_argument("--save_debug_vis", action="store_true",
                        help="是否保存调试可视化")

    # 数据集图像根目录
    parser.add_argument("--lasot_root", type=str, 
                        default="/home/member/data1/DATASETS_PUBLIC/LaSOT/LaSOTBenchmark",
                        help="LaSOT 数据集图像根目录")
    parser.add_argument("--mgit_root", type=str, 
                        default="/home/member/data1/DATASETS_PUBLIC/MGIT/VideoCube/MGIT-Test/data/test",
                        help="MGIT 数据集图像根目录")
    parser.add_argument("--tnl2k_root", type=str, 
                        default="/home/member/data1/DATASETS_PUBLIC/TNL2K/TNL2K_CVPR2021/test",
                        help="TNL2K 数据集图像根目录")

    # JSONL 描述文件目录
    parser.add_argument("--lasot_jsonl", type=str, 
                        default="/home/member/data2/wyp/SUTrack/SOIBench/data/test/lasot",
                        help="LaSOT JSONL 描述文件目录")
    parser.add_argument("--mgit_jsonl", type=str, 
                        default="/home/member/data2/wyp/SUTrack/SOIBench/data/test/mgit",
                        help="MGIT JSONL 描述文件目录")
    parser.add_argument("--tnl2k_jsonl", type=str, 
                        default="/home/member/data2/wyp/SUTrack/SOIBench/data/test/tnl2k",
                        help="TNL2K JSONL 描述文件目录")

    # 输出目录
    parser.add_argument("--output_root", type=str, default="./results",
                        help="结果保存根目录")

    args = parser.parse_args()

    # 构建图像根目录字典
    image_roots = {
        "lasot": args.lasot_root,
        "mgit": args.mgit_root,
        "tnl2k": args.tnl2k_root
    }

    # 初始化推理引擎
    engine = None
    if args.mode == "local":
        if not args.model_path:
            raise ValueError("mode=local 时必须提供 --model_path")
        print(f"🚀 加载本地模型: {args.model_path}")
        engine = Qwen3VLLocalEngine(args.model_path)

    # 构建任务列表
    tasks = []
    if args.lasot_jsonl and os.path.exists(args.lasot_jsonl):
        tasks.append(("lasot", args.lasot_jsonl))
    if args.mgit_jsonl and os.path.exists(args.mgit_jsonl):
        tasks.append(("mgit", args.mgit_jsonl))
    if args.tnl2k_jsonl and os.path.exists(args.tnl2k_jsonl):
        tasks.append(("tnl2k", args.tnl2k_jsonl))

    if not tasks:
        print("❌ 未指定任何有效的数据目录")
        return

    # 处理每个数据集
    for dataset_name, jsonl_dir in tasks:
        out_dir = os.path.join(args.output_root, dataset_name, f"{args.mode}_{args.exp_tag}")
        os.makedirs(out_dir, exist_ok=True)

        vis_dir = None
        if args.save_debug_vis:
            vis_dir = os.path.join(out_dir, "vis_debug")
            os.makedirs(vis_dir, exist_ok=True)

        jsonl_files = sorted(glob.glob(os.path.join(jsonl_dir, "*.jsonl")))
        if not jsonl_files:
            print(f"⚠️  目录为空: {jsonl_dir}")
            continue

        print(f"\n📂 处理数据集: {dataset_name} ({len(jsonl_files)} 个序列)")

        for jsonl_file in tqdm(jsonl_files, desc=f"处理 {dataset_name}", dynamic_ncols=True):
            seq_name = Path(jsonl_file).stem.replace("_descriptions", "").replace("_done", "")
            save_path = os.path.join(out_dir, f"{seq_name}_pred.jsonl")

            # 断点续跑: 检查已处理的行数
            processed = _count_lines(save_path)
            samples = load_and_fix_paths(jsonl_file, dataset_name, image_roots)

            if processed >= len(samples):
                continue

            pending = samples[processed:]

            # 创建输出文件
            if not os.path.exists(save_path):
                with open(save_path, "w", encoding="utf-8"):
                    pass

            # 处理每一帧
            for s in tqdm(pending, desc=f"序列 {seq_name}", leave=False, dynamic_ncols=True):
                img_path = s["image_path"]
                
                prompt = build_prompt(s["text_prompt"])

                # 调用推理引擎
                if args.mode == "local":
                    raw_out = engine.chat(img_path, prompt, max_new_tokens=args.max_new_tokens)
                else:
                    raw_out = qwen3vl_api_chat(
                        image_path=img_path,
                        prompt=prompt,
                        model_name=args.api_model_name,
                        base_url=args.api_base_url,
                        api_key=args.api_key_env,
                        temperature=args.api_temperature,
                        max_tokens=args.api_max_tokens,
                        retries=args.api_retries,
                        retry_sleep=1.0,
                    )

                # 处理空输出
                if not raw_out:
                    record = s["original_item"].copy()
                    record["model_raw_response"] = raw_out
                    record["parsed_bboxes"] = []
                    record["parse_status"] = "empty_output"
                    with open(save_path, "a", encoding="utf-8") as f_out:
                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue

                # 解析 bbox
                with Image.open(img_path) as img:
                    w, h = img.size
                    parsed = extract_bboxes_from_model_output(raw_out, w, h)

                # 保存结果
                record = s["original_item"].copy()
                record["model_raw_response"] = raw_out
                record["parsed_bboxes"] = parsed
                record["parse_status"] = "ok" if parsed else "no_bbox_found"

                with open(save_path, "a", encoding="utf-8") as f_out:
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

                # 可选: 保存可视化
                if vis_dir and parsed:
                    vis_path = os.path.join(vis_dir, f"{seq_name}_{s['frame_idx']}.jpg")
                    with Image.open(img_path) as img:
                        plot_bounding_boxes(img, parsed, vis_path)

    print("\n✅ 全部任务完成")


if __name__ == "__main__":
    main()

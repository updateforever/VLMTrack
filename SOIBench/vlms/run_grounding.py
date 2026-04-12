# -*- coding: utf-8 -*-
"""
SOIBench/vlms/run_grounding.py
统一的 Grounding 推理入口脚本
支持多种 VLM 模型通过适配器接入

使用方法:
    # Qwen3VL API 推理
    python run_grounding.py --model qwen3vl --mode api
    
    # GLM-4.6V 本地推理
    python run_grounding.py --model glm46v --mode local --model_path /path/to/model
    
    # DeepSeek-VL2 API 推理
    python run_grounding.py --model deepseekvl --mode api
    
    # 添加新模型
    from model_adapters import register_adapter, ModelAdapter
    class MyVLMAdapter(ModelAdapter):
        ...
    register_adapter('myvlm', MyVLMAdapter)
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageColor

from model_adapters import get_adapter


# ============================================================================
# 通用辅助函数
# ============================================================================

_ADDITIONAL_COLORS = [name for (name, _) in ImageColor.colormap.items()]


def _pick_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _resolve_default_paths():
    lasot_jsonl = _pick_existing([
        os.environ.get('SOIBENCH_LASOT_JSONL'),
        '/root/user-data/wyp/VLMTrack/SOIBench/data/test/lasot',
        '/home/member/data2/wyp/SUTrack/SOIBench/data/test/lasot',
    ]) or ''
    mgit_jsonl = _pick_existing([
        os.environ.get('SOIBENCH_MGIT_JSONL'),
        '/root/user-data/wyp/VLMTrack/SOIBench/data/test/mgit',
        '/home/member/data2/wyp/SUTrack/SOIBench/data/test/mgit',
    ]) or ''
    tnl2k_jsonl = _pick_existing([
        os.environ.get('SOIBENCH_TNL2K_JSONL'),
        '/root/user-data/wyp/VLMTrack/SOIBench/data/test/tnl2k',
        '/home/member/data2/wyp/SUTrack/SOIBench/data/test/tnl2k',
    ]) or ''

    lasot_root = _pick_existing([
        os.environ.get('LASOT_ROOT'),
        '/root/user-data/DATASETS_PUBLIC/LaSOT/LaSOTBenchmark',
        '/home/member/data1/DATASETS_PUBLIC/LaSOT/LaSOTBenchmark',
    ]) or ''
    mgit_root = _pick_existing([
        os.environ.get('MGIT_ROOT'),
        '/root/user-data/DATASETS_PUBLIC/MGIT',
        '/home/member/data1/DATASETS_PUBLIC/MGIT/VideoCube/data/test',
        '/home/member/data1/DATASETS_PUBLIC/MGIT/VideoCube/MGIT-Test/data/test',
    ]) or ''
    tnl2k_root = _pick_existing([
        os.environ.get('TNL2K_ROOT'),
        '/root/user-data/DATASETS_PUBLIC/TNL2K/TNL2K_test_subset',
        '/home/member/data1/DATASETS_PUBLIC/TNL2K_test/TNL2K_test_subset',
    ]) or ''
    return lasot_jsonl, mgit_jsonl, tnl2k_jsonl, lasot_root, mgit_root, tnl2k_root


def plot_bounding_boxes(im: Image.Image, bboxes: List[List[float]], save_path: str):
    """在图上画 bbox"""
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
    """处理描述文本的四个层级，添加合适的标点"""
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
    """读取描述 jsonl，并把 image_path 修复为绝对路径"""
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
    """统计文件行数，用于断点续跑"""
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
    """运行 Grounding 推理的主流程"""
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


# ============================================================================
# 主函数
# ============================================================================

def main():
    (default_lasot_jsonl,
     default_mgit_jsonl,
     default_tnl2k_jsonl,
     default_lasot_root,
     default_mgit_root,
     default_tnl2k_root) = _resolve_default_paths()

    parser = argparse.ArgumentParser(
        description="SOIBench Grounding 推理统一入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # Qwen3VL API 推理
  python run_grounding.py --model qwen3vl --mode api
  
  # GLM-4.6V 本地推理  
  python run_grounding.py --model glm46v --mode local
  
  # DeepSeek-VL2 API 推理
  python run_grounding.py --model deepseekvl --mode api
        """
    )
    
    # 模型选择
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3vl", "glm46v", "deepseekvl"],
                        help="模型名称")
    
    # 推理模式
    parser.add_argument("--mode", type=str, required=True,
                        choices=["local", "api"],
                        help="推理模式: local (本地模型) 或 api (API)")
    
    # 本地模型参数
    parser.add_argument("--model_path", type=str, default=None,
                        help="本地模型路径 (mode=local 时使用，不指定则使用默认路径)")
    
    # API 参数
    parser.add_argument("--api_key", type=str, default=None,
                        help="API Key (默认从环境变量读取)")
    parser.add_argument("--api_model_name", type=str, default=None,
                        help="API 模型名称 (不指定则使用默认值)")
    parser.add_argument("--api_base_url", type=str, default=None,
                        help="API Base URL (不指定则使用默认值)")
    parser.add_argument("--api_temperature", type=float, default=0.1,
                        help="API 温度参数")
    parser.add_argument("--api_max_tokens", type=int, default=512,
                        help="API 最大 token 数")
    parser.add_argument("--api_retries", type=int, default=3,
                        help="API 重试次数")
    
    # 数据集参数
    parser.add_argument("--lasot_jsonl", type=str,
                        default=default_lasot_jsonl,
                        help="LaSOT JSONL 描述文件目录")
    parser.add_argument("--lasot_root", type=str,
                        default=default_lasot_root,
                        help="LaSOT 图像根目录")
    parser.add_argument("--mgit_jsonl", type=str,
                        default=default_mgit_jsonl,
                        help="MGIT JSONL 描述文件目录")
    parser.add_argument("--mgit_root", type=str,
                        default=default_mgit_root,
                        help="MGIT 图像根目录")
    parser.add_argument("--tnl2k_jsonl", type=str,
                        default=default_tnl2k_jsonl,
                        help="TNL2K JSONL 描述文件目录")
    parser.add_argument("--tnl2k_root", type=str,
                        default=default_tnl2k_root,
                        help="TNL2K 图像根目录")
    
    # 输出参数
    parser.add_argument("--output_root", type=str, default="./SOIBench/results",
                        help="输出根目录")
    parser.add_argument("--exp_tag", type=str, default=None,
                        help="实验标签 (不指定则使用模型名)")
    parser.add_argument("--save_debug_vis", action="store_true",
                        help="是否保存调试可视化图像")
    
    # 推理参数
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="最大生成 token 数")
    
    args = parser.parse_args()
    
    # 获取适配器
    print(f"🔧 加载模型适配器: {args.model}")
    adapter_class = get_adapter(args.model)
    adapter = adapter_class()
    
    # 设置默认值
    if args.mode == 'local' and not args.model_path:
        args.model_path = adapter.get_default_model_path()
        if not args.model_path:
            raise ValueError(f"模型 {args.model} 没有默认本地路径，请使用 --model_path 指定")
        print(f"📁 使用默认模型路径: {args.model_path}")
    
    if args.mode == 'api':
        if not args.api_model_name:
            args.api_model_name = adapter.get_default_api_model_name()
            print(f"🔤 使用默认 API 模型名: {args.api_model_name}")
        if not args.api_base_url:
            args.api_base_url = adapter.get_default_api_base_url()
            print(f"🌐 使用默认 API Base URL: {args.api_base_url}")
    
    if not args.exp_tag:
        args.exp_tag = args.model
    
    # 创建推理引擎
    print(f"🚀 初始化推理引擎 (mode={args.mode})")
    engine = adapter.create_engine(args)
    
    # 构建图像根目录字典
    image_roots = {
        "lasot": args.lasot_root,
        "mgit": args.mgit_root,
        "tnl2k": args.tnl2k_root
    }
    
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
        
        print(f"\n{'='*60}")
        print(f"处理数据集: {dataset_name}")
        print(f"JSONL 目录: {jsonl_dir}")
        print(f"图像根目录: {image_roots[dataset_name]}")
        print(f"输出目录: {out_dir}")
        print(f"{'='*60}")
        
        # 运行推理
        run_grounding_inference(
            adapter=adapter,
            engine=engine,
            jsonl_dir=jsonl_dir,
            dataset_name=dataset_name,
            image_roots=image_roots,
            output_dir=out_dir,
            vis_dir=vis_dir,
            max_new_tokens=args.max_new_tokens,
        )
    
    print("\n✅ 全部任务完成")


if __name__ == "__main__":
    main()

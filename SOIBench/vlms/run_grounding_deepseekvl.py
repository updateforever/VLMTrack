# -*- coding: utf-8 -*-
"""
SOIBench/vlms/run_grounding_deepseekvl.py
DeepSeek-VL2 Grounding 推理脚本
适配 SOIBench 数据集
"""

import argparse
import json
import os
from tqdm import tqdm
from PIL import Image

from deepseekvl_infer import DeepSeekVLLocalEngine, DeepSeekVLAPIEngine, parse_deepseekvl_bbox
import cv2
import numpy as np


def draw_bbox_on_image(image_path, gt_box, pred_boxes, output_path):
    """
    在图像上绘制 GT 和预测的 bbox
    
    参数:
        image_path: 图像路径
        gt_box: GT bbox [[x1, y1], [x2, y2]]
        pred_boxes: 预测 bbox 列表 [[x1, y1, x2, y2], ...]
        output_path: 输出路径
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        return
    
    # 绘制 GT (绿色)
    if gt_box and len(gt_box) == 2:
        x1, y1 = int(gt_box[0][0]), int(gt_box[0][1])
        x2, y2 = int(gt_box[1][0]), int(gt_box[1][1])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, 'GT', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 绘制预测 (红色)
    for idx, pred_box in enumerate(pred_boxes):
        if len(pred_box) == 4:
            x1, y1, x2, y2 = map(int, pred_box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f'Pred{idx+1}', (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # 保存
    cv2.imwrite(output_path, img)


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-VL2 Grounding 推理脚本")
    
    # 模式选择
    parser.add_argument("--mode", type=str, required=True, choices=["local", "api"],
                        help="推理模式: local (本地模型) 或 api (API)")
    
    # 本地模型参数
    parser.add_argument("--model_path", type=str, 
                        default="/home/member/data1/MODEL_WEIGHTS_PUBLIC/deepseek-vl2-small/",
                        help="本地模型路径 (mode=local 时使用)")
    
    # API 参数
    parser.add_argument("--api_key", type=str, default=None,
                        help="API Key (默认从环境变量 SILICONFLOW_API_KEY 读取)")
    parser.add_argument("--api_model_name", type=str, default="deepseek-ai/deepseek-vl2",
                        help="API 模型名称")
    parser.add_argument("--api_base_url", type=str, 
                        default="https://api.siliconflow.cn/v1",
                        help="API Base URL")
    parser.add_argument("--api_temperature", type=float, default=0.1,
                        help="API 温度参数")
    parser.add_argument("--api_max_tokens", type=int, default=512,
                        help="API 最大 token 数")
    parser.add_argument("--api_retries", type=int, default=3,
                        help="API 重试次数")
    
    # 数据集参数 (使用默认路径)
    parser.add_argument("--lasot_jsonl", type=str,
                        default="/home/member/data2/wyp/SUTrack/SOIBench/data/test/lasot",
                        help="LaSOT JSONL 描述文件目录")
    parser.add_argument("--lasot_root", type=str,
                        default="/home/member/data1/DATASETS_PUBLIC/LaSOT/LaSOTBenchmark",
                        help="LaSOT 图像根目录")
    parser.add_argument("--mgit_jsonl", type=str,
                        default="/home/member/data2/wyp/SUTrack/SOIBench/data/test/mgit",
                        help="MGIT JSONL 描述文件目录")
    parser.add_argument("--mgit_root", type=str,
                        default="/home/member/data1/DATASETS_PUBLIC/MGIT/VideoCube/data/test",
                        help="MGIT 图像根目录")
    parser.add_argument("--tnl2k_jsonl", type=str,
                        default="/home/member/data2/wyp/SUTrack/SOIBench/data/test/tnl2k",
                        help="TNL2K JSONL 描述文件目录")
    parser.add_argument("--tnl2k_root", type=str,
                        default="/home/member/data1/DATASETS_PUBLIC/TNL2K_test/TNL2K_test_subset",
                        help="TNL2K 图像根目录")
    
    # 输出参数
    parser.add_argument("--output_root", type=str, default="./SOIBench/results",
                        help="输出根目录")
    parser.add_argument("--exp_tag", type=str, default="deepseekvl",
                        help="实验标签")
    parser.add_argument("--save_debug_vis", action="store_true",
                        help="是否保存调试可视化图像")
    
    args = parser.parse_args()
    
    # 初始化推理引擎
    if args.mode == "local":
        if not args.model_path:
            raise ValueError("mode=local 时必须提供 --model_path")
        engine = DeepSeekVLLocalEngine(args.model_path)
    else:  # api
        engine = DeepSeekVLAPIEngine(
            api_key=args.api_key,
            api_base_url=args.api_base_url,
            model_name=args.api_model_name,
            temperature=args.api_temperature,
            max_tokens=args.api_max_tokens,
            retries=args.api_retries,
        )
    
    # 数据集配置
    datasets = {
        "lasot": {
            "jsonl_dir": args.lasot_jsonl,
            "image_root": args.lasot_root
        },
        "mgit": {
            "jsonl_dir": args.mgit_jsonl,
            "image_root": args.mgit_root
        },
        "tnl2k": {
            "jsonl_dir": args.tnl2k_jsonl,
            "image_root": args.tnl2k_root
        }
    }
    
    # 处理每个数据集
    for dataset_name, config in datasets.items():
        jsonl_dir = config["jsonl_dir"]
        image_root = config["image_root"]
        
        if not os.path.exists(jsonl_dir):
            print(f"⚠️  跳过 {dataset_name}: 目录不存在 {jsonl_dir}")
            continue
        
        # 输出目录
        output_dir = os.path.join(args.output_root, dataset_name, f"{args.mode}_{args.exp_tag}")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"处理数据集: {dataset_name}")
        print(f"JSONL 目录: {jsonl_dir}")
        print(f"图像根目录: {image_root}")
        print(f"输出目录: {output_dir}")
        print(f"{'='*60}\n")
        
        # 获取所有 JSONL 文件
        jsonl_files = [f for f in os.listdir(jsonl_dir) if f.endswith('_descriptions.jsonl')]
        
        for jsonl_file in tqdm(jsonl_files, desc=f"处理 {dataset_name}"):
            seq_name = jsonl_file.replace('_descriptions.jsonl', '')
            jsonl_path = os.path.join(jsonl_dir, jsonl_file)
            output_path = os.path.join(output_dir, f"{seq_name}_pred.jsonl")
            
            # 可视化目录
            vis_dir = None
            if args.save_debug_vis:
                vis_dir = os.path.join(output_dir, f"{seq_name}_vis")
                os.makedirs(vis_dir, exist_ok=True)
            
            # 如果已经处理过，跳过
            if os.path.exists(output_path):
                print(f"  ⏭️  跳过已处理: {seq_name}")
                continue
            
            # 读取 JSONL
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 处理每一帧
            results = []
            for line in lines:
                if not line.strip():
                    continue
                
                item = json.loads(line)
                frame_idx = item.get('frame_idx')
                status = item.get('status', '')
                image_path = item.get('image_path', '')
                
                # 构造完整图像路径
                if image_path.startswith('/'):
                    image_path = image_path[1:]
                
                # 尝试多种路径组合
                full_image_path = None
                possible_paths = [
                    os.path.join(image_root, image_path),
                    os.path.join(image_root, image_path[6:10], image_path) if len(image_path) > 10 else None,
                ]
                
                if len(image_path.split('/')) > 2:
                    parts = image_path.split('/')
                    possible_paths.append(os.path.join(image_root, parts[1], 'imgs', parts[2][1:]))
                    possible_paths.append(os.path.join(image_root, parts[1], 'imgs', parts[2]))
                
                for p in possible_paths:
                    if p and os.path.exists(p):
                        full_image_path = p
                        break
                
                if not full_image_path:
                    print(f"  ⚠️  图像未找到: {image_path}")
                    results.append({
                        **item,
                        "model_response": "",
                        "parsed_bboxes": []
                    })
                    continue
                
                # 构造 prompt
                # 注意：即使是 skip 帧，VLM 也需要推理！
                # skip 只是人类标注时跳过，算法需要对所有帧都预测
                output_en = item.get("output-en", {}) or {}
                
                # 构造分层描述，添加合适的标点
                # Level 1, 2: 逗号结尾
                # Level 3, 4: 句号结尾
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
                
                prompt = " ".join(desc_parts).strip()
                if not prompt:
                    print(f"  ⚠️  WARNING: 序列 {seq_name} 的帧 {frame_idx} 缺少描述文本，使用默认 prompt")
                    prompt = "the target object"
                
                # 推理
                try:
                    response = engine.chat(full_image_path, prompt)
                    
                    # 解析 bbox
                    img = Image.open(full_image_path)
                    bboxes = parse_deepseekvl_bbox(response, img.width, img.height)
                    
                    results.append({
                        **item,
                        "model_response": response,
                        "parsed_bboxes": bboxes
                    })
                    
                    # 保存可视化
                    if vis_dir:
                        vis_path = os.path.join(vis_dir, f"{frame_idx:08d}.jpg")
                        gt_box = item.get("gt_box")
                        draw_bbox_on_image(full_image_path, gt_box, bboxes, vis_path)
                    
                except Exception as e:
                    print(f"  ❌ 推理失败 ({seq_name}, frame {frame_idx}): {e}")
                    results.append({
                        **item,
                        "model_response": f"ERROR: {str(e)}",
                        "parsed_bboxes": []
                    })
            
            # 保存结果
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            print(f"  ✅ 完成: {seq_name} ({len(results)} 帧)")
    
    print(f"\n{'='*60}")
    print("🎉 所有数据集处理完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

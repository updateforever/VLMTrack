# -*- coding: utf-8 -*-
"""
SOIBench/vlms/visualize_grounding.py
可视化 Grounding 结果
功能：
1）读取多个模型的 Pred JSONL 和 GT JSONL
2）在原图上画 GT (绿色)、多个模型预测 (不同颜色)、人类基线 (蓝色)
3）保存为图片或视频
"""

import argparse
import json
import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont


# 为不同模型分配颜色
MODEL_COLORS = [
    "red", "orange", "purple", "magenta", "cyan", 
    "yellow", "pink", "brown", "gray", "olive"
]


def load_seq_data(jsonl_path, is_gt=False, load_human_baseline=False):
    """
    加载单个序列文件，返回 {frame_idx: (box, image_path)} 字典
    参数:
        jsonl_path: JSONL 文件路径
        is_gt: 是否为 GT 文件
        load_human_baseline: 是否加载人类基线
    返回:
        {frame_idx: (box, image_path)} 字典
    """
    data_map = {}
    if not os.path.exists(jsonl_path):
        return data_map

    last_valid_box = None  # 用于 skip 帧的填充
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                
                fid = int(item.get("frame_idx", -1))
                if fid == -1:
                    continue
                
                is_skip = item.get("status") == "skip"
                img_path = item.get("image_path", "")

                if load_human_baseline:
                    # 加载人类基线: 从 pred_boxes 提取
                    pred_boxes = item.get("pred_boxes", [])
                    
                    if is_skip:
                        # skip 帧: 使用上一个有效帧的结果
                        if last_valid_box is not None:
                            data_map[fid] = (last_valid_box, img_path)
                    else:
                        # 非 skip 帧: 提取 pred_boxes
                        if pred_boxes and len(pred_boxes) > 0:
                            # pred_boxes 格式: [[x1,y1], [x2,y2]] -> [x1,y1,x2,y2]
                            box = pred_boxes
                            if len(box) == 2 and len(box[0]) == 2:
                                box = [box[0][0], box[0][1], box[1][0], box[1][1]]
                            last_valid_box = box
                            data_map[fid] = (box, img_path)
                
                elif is_gt:
                    # GT 提取逻辑
                    if not is_skip:
                        box = item.get("gt_box") or item.get("bbox")
                        # gt_box 格式: [[x1,y1], [x2,y2]] -> [x1,y1,x2,y2]
                        if box and len(box) == 2 and len(box[0]) == 2:
                            box = [box[0][0], box[0][1], box[1][0], box[1][1]]
                        if box:
                            data_map[fid] = (box, img_path)
                else:
                    # Pred 提取逻辑: 取第一个预测框
                    p_boxes = item.get("parsed_bboxes") or item.get("parsed_bbox")
                    box = p_boxes[0] if (p_boxes and len(p_boxes) > 0) else None
                    if box:
                        data_map[fid] = (box, img_path)
                        
            except:
                continue
    return data_map


def draw_box(img, box, color, label=None):
    """
    在图像上绘制 bbox
    参数:
        img: PIL Image 对象
        box: [x1, y1, x2, y2]
        color: 颜色
        label: 标签文本
    返回:
        绘制后的 PIL Image
    """
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=20)
    except:
        font = ImageFont.load_default()
        
    x1, y1, x2, y2 = [int(v) for v in box]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    if label:
        # 绘制标签背景
        text_bbox = draw.textbbox((x1, y1-25), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1-25), label, fill="white", font=font)
    return img


def fix_image_path(rel_path, image_root):
    """
    修复图像路径
    参数:
        rel_path: 相对路径
        image_root: 图像根目录
    返回:
        绝对路径
    """
    if not rel_path:
        return None
        
    if rel_path.startswith("/"):
        rel_path = rel_path[1:]
    
    # 尝试多种路径组合
    possible_paths = [
        rel_path if os.path.isabs(rel_path) else None,  # 已经是绝对路径
        os.path.join(image_root, rel_path),
    ]
    
    # LaSOT 特殊路径
    if len(rel_path) > 10:
        possible_paths.append(os.path.join(image_root, rel_path[6:10], rel_path))
    
    # MGIT/TNL2K 特殊路径
    if len(rel_path.split('/')) > 2:
        parts = rel_path.split('/')
        possible_paths.append(os.path.join(image_root, parts[1], 'imgs', parts[2][1:]))
        possible_paths.append(os.path.join(image_root, parts[1], 'imgs', parts[2]))
    
    for p in possible_paths:
        if p and os.path.exists(p):
            return p
    
    return None


def main():
    parser = argparse.ArgumentParser(description="SOIBench Grounding 可视化脚本")
    
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["lasot", "mgit", "tnl2k"],
                        help="数据集名称")
    parser.add_argument("--seq_name", type=str, required=True,
                        help="序列名称")
    parser.add_argument("--pred_root", type=str, required=True,
                        help="预测结果根目录")
    parser.add_argument("--models", nargs='+', required=True,
                        help="要可视化的模型 tag 列表，例如: model1 model2")
    parser.add_argument("--gt_file", type=str, required=True,
                        help="GT JSONL 文件路径")
    parser.add_argument("--image_root", type=str, required=True,
                        help="图片根目录")
    parser.add_argument("--output_dir", type=str, default="./vis_results",
                        help="可视化保存目录")
    parser.add_argument("--save_video", action="store_true",
                        help="是否保存为视频")
    parser.add_argument("--fps", type=int, default=30,
                        help="视频帧率")
    parser.add_argument("--show_human_baseline", action="store_true",
                        help="是否显示人类基线 (从 GT JSONL 的 pred_boxes 提取)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n📂 可视化序列: {args.seq_name}")
    print(f"📊 模型数量: {len(args.models)}")
    
    # 加载 GT 数据
    gt_map = load_seq_data(args.gt_file, is_gt=True)
    
    # 加载人类基线 (如果需要)
    human_map = {}
    if args.show_human_baseline:
        human_map = load_seq_data(args.gt_file, is_gt=False, load_human_baseline=True)
        print(f"📊 人类基线: {len(human_map)} 帧")
    
    # 加载所有模型的预测结果
    model_maps = {}
    for model_tag in args.models:
        # 尝试多种预测文件路径结构
        possible_paths = [
            os.path.join(args.pred_root, args.dataset, model_tag, f"{args.seq_name}_pred.jsonl"),
            os.path.join(args.pred_root, args.dataset, f"{args.seq_name}_{model_tag}_pred.jsonl")
        ]
        
        pred_file = None
        for p in possible_paths:
            if os.path.exists(p):
                pred_file = p
                break
        
        if pred_file:
            model_maps[model_tag] = load_seq_data(pred_file, is_gt=False)
            print(f"✅ 加载模型 {model_tag}: {len(model_maps[model_tag])} 帧")
        else:
            print(f"⚠️  未找到模型 {model_tag} 的预测文件")
            model_maps[model_tag] = {}
    
    # 获取所有帧索引
    all_fids = set(gt_map.keys()) | set(human_map.keys())
    for model_map in model_maps.values():
        all_fids |= set(model_map.keys())
    all_fids = sorted(list(all_fids))
    
    if not all_fids:
        print("❌ 没有找到任何帧数据")
        return

    print(f"📊 共 {len(all_fids)} 帧")
    
    # 准备视频写入器
    video_writer = None
    if args.save_video:
        video_path = os.path.join(args.output_dir, f"{args.seq_name}_compare.mp4")

    # 处理每一帧
    for fid in tqdm(all_fids, desc="可视化"):
        # 获取图像路径
        img_path = None
        if fid in gt_map:
            _, img_path = gt_map[fid]
        elif fid in human_map:
            _, img_path = human_map[fid]
        else:
            for model_map in model_maps.values():
                if fid in model_map:
                    _, img_path = model_map[fid]
                    break
        
        # 修复图像路径
        if img_path:
            img_path = fix_image_path(img_path, args.image_root)
        
        if not img_path or not os.path.exists(img_path):
            continue
            
        # 读取图像
        img = Image.open(img_path).convert("RGB")
        
        # 画 GT (绿色)
        if fid in gt_map:
            gt_box, _ = gt_map[fid]
            img = draw_box(img, gt_box, "green", "GT")
        
        # 画人类基线 (蓝色)
        if args.show_human_baseline and fid in human_map:
            human_box, _ = human_map[fid]
            img = draw_box(img, human_box, "blue", "Human")
        
        # 画所有模型的预测 (不同颜色)
        for idx, (model_tag, model_map) in enumerate(model_maps.items()):
            if fid in model_map:
                pred_box, _ = model_map[fid]
                color = MODEL_COLORS[idx % len(MODEL_COLORS)]
                img = draw_box(img, pred_box, color, model_tag)
            
        # 保存图片
        if not args.save_video:
            save_path = os.path.join(args.output_dir, args.seq_name, f"{fid:08d}.jpg")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            img.save(save_path)
        else:
            # 转换为 OpenCV 格式写入视频
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            if video_writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_path, fourcc, args.fps, (w, h))
            video_writer.write(frame)

    # 释放视频写入器
    if video_writer:
        video_writer.release()
        print(f"✅ 视频已保存: {video_path}")
    else:
        print(f"✅ 图片已保存至: {os.path.join(args.output_dir, args.seq_name)}")


if __name__ == "__main__":
    main()

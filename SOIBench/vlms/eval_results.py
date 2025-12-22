# -*- coding: utf-8 -*-
"""
SOIBench/vlms/eval_results.py
评测 Grounding 结果
功能：
1）加载 GT 和 Pred JSONL
2）计算 IoU 指标
3）绘制 Success Plot
4）生成评测报告
"""

import argparse
import glob
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from tqdm import tqdm


def calculate_iou(box1, box2):
    """
    计算两个 bbox 的 IoU
    参数:
        box1, box2: [x1, y1, x2, y2] 格式的 bbox
    返回:
        IoU 值 (0-1)
    """
    if not box1 or not box2:
        return 0.0
    
    # 兼容可能出现的嵌套 list
    b1 = [float(x) for x in (box1[0] if isinstance(box1[0], list) else box1)]
    b2 = [float(x) for x in (box2[0] if isinstance(box2[0], list) else box2)]
    
    # 计算交集
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    
    # 计算并集
    area1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    area2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = area1 + area2 - inter_area
    
    return inter_area / union if union > 0 else 0.0


def load_seq_data(jsonl_path, is_gt=False):
    """
    加载单个序列文件，返回 {frame_idx: box} 字典
    参数:
        jsonl_path: JSONL 文件路径
        is_gt: 是否为 GT 文件
    返回:
        {frame_idx: box} 字典
    """
    data_map = {}
    if not os.path.exists(jsonl_path):
        return data_map

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                
                # 跳过 skip 的帧 (仅针对 GT)
                if is_gt and item.get("status") == "skip":
                    continue
                
                fid = int(item.get("frame_idx", -1))
                if fid == -1:
                    continue

                if is_gt:
                    # GT 提取逻辑
                    box = item.get("gt_box") or item.get("bbox")
                else:
                    # Pred 提取逻辑: 取第一个预测框
                    p_boxes = item.get("parsed_bboxes") or item.get("parsed_bbox")
                    box = p_boxes[0] if (p_boxes and len(p_boxes) > 0) else None
                
                if box:
                    data_map[fid] = box
            except:
                continue
    return data_map


def evaluate_dataset(ds_name, gt_root, pred_root, model_tags):
    """
    评测单个数据集
    参数:
        ds_name: 数据集名称
        gt_root: GT JSONL 文件根目录
        pred_root: 预测结果根目录
        model_tags: 模型标签列表
    返回:
        {model_name: [all_ious_list]} 字典
    """
    # 查找该数据集所有 GT 文件
    gt_files = sorted(glob.glob(os.path.join(gt_root, "*.jsonl")))
    if not gt_files:
        print(f"⚠️  {ds_name} 未找到 GT 文件，跳过。")
        return {}

    # 存储每个模型在该数据集下的所有帧 IoU
    model_ious = {tag: [] for tag in model_tags}
    
    print(f"🔄 正在评测 {ds_name} ({len(gt_files)} 个序列)...")

    for gt_path in tqdm(gt_files, leave=False, desc=f"评测 {ds_name}"):
        seq_name = os.path.basename(gt_path).replace("_descriptions.jsonl", "").replace(".jsonl", "")
        
        # 加载 GT {frame: box}
        gt_map = load_seq_data(gt_path, is_gt=True)
        if not gt_map:
            continue

        # 遍历每个模型
        for tag in model_tags:
            # 尝试多种预测文件路径结构
            # 1. pred_root/dataset_name/tag/seq_name_pred.jsonl (新结构)
            # 2. pred_root/dataset_name/seq_name_tag_pred.jsonl (旧结构)
            
            possible_paths = [
                os.path.join(pred_root, ds_name, tag, f"{seq_name}_pred.jsonl"),
                os.path.join(pred_root, ds_name, f"{seq_name}_{tag}_pred.jsonl")
            ]
            
            pred_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    pred_path = p
                    break
            
            # 加载 Pred {frame: box}
            pred_map = load_seq_data(pred_path, is_gt=False) if pred_path else {}
            
            # 逐帧对齐计算 (以 GT 为准)
            for fid, gt_box in gt_map.items():
                pred_box = pred_map.get(fid)
                iou = calculate_iou(pred_box, gt_box)
                model_ious[tag].append(iou)
                
    return model_ious


def plot_success_curves(results, output_dir, ds_name):
    """
    绘制 Success Plot (成功率曲线)
    参数:
        results: {model_name: [iou_list]} 字典
        output_dir: 输出目录
        ds_name: 数据集名称
    """
    plt.figure(figsize=(10, 7))
    plt.title(f"Success Plot - {ds_name.upper()}", fontsize=16)
    plt.xlabel("Overlap Threshold (IoU)", fontsize=14)
    plt.ylabel("Success Rate", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    thresholds = np.linspace(0, 1, 21)  # 0, 0.05, ..., 1.0
    
    # 按 AUC 排序图例
    model_stats = []
    
    for model_name, ious in results.items():
        if not ious:
            continue
        ious_arr = np.array(ious)
        
        # 计算曲线点: 每个阈值下的成功率
        curve = [np.mean(ious_arr >= thr) for thr in thresholds]
        auc = np.mean(curve)  # 近似 AUC
        model_stats.append((model_name, auc, curve))
        
    # 按 AUC 降序排序
    model_stats.sort(key=lambda x: x[1], reverse=True)
    
    # 绘制曲线
    for model_name, auc, curve in model_stats:
        plt.plot(thresholds, curve, linewidth=2, label=f"{model_name} [{auc:.3f}]")
        
    plt.legend(loc='lower left', fontsize=12)
    
    save_path = os.path.join(output_dir, f"{ds_name}_success_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 曲线图已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="SOIBench Grounding 评测脚本")
    
    parser.add_argument("--pred_root", type=str, default="./results",
                        help="预测结果根目录")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="评测结果保存目录")
    parser.add_argument("--models", nargs='+', required=True,
                        help="要对比的模型 tag 列表，例如: local_run api_run")
    parser.add_argument("--datasets", nargs='+', default=["lasot", "mgit", "tnl2k"],
                        help="要评测的数据集")
    
    # GT 根目录
    parser.add_argument("--lasot_gt_root", type=str, 
                        default="/home/member/data2/wyp/SUTrack/SOIBench/data/test/lasot",
                        help="LaSOT GT JSONL 根目录")
    parser.add_argument("--mgit_gt_root", type=str, 
                        default="/home/member/data2/wyp/SUTrack/SOIBench/data/test/mgit",
                        help="MGIT GT JSONL 根目录")
    parser.add_argument("--tnl2k_gt_root", type=str, 
                        default="/home/member/data2/wyp/SUTrack/SOIBench/data/test/tnl2k",
                        help="TNL2K GT JSONL 根目录")

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 构建 GT 根目录字典
    gt_roots = {
        "lasot": args.lasot_gt_root,
        "mgit": args.mgit_gt_root,
        "tnl2k": args.tnl2k_gt_root
    }
    
    # 创建总表
    summary_table = PrettyTable()
    summary_table.field_names = ["Dataset", "Model", "AUC", "OP@0.50", "OP@0.75"]
    
    print("\n" + "="*60)
    print("SOIBench Grounding 评测")
    print("="*60)
    
    for ds_name in args.datasets:
        gt_root = gt_roots.get(ds_name)
        
        if not gt_root or not os.path.exists(gt_root):
            print(f"❌ 无法找到 {ds_name} 的 GT 目录: {gt_root}，跳过")
            continue
        
        # 计算该数据集下所有模型的 IoU
        dataset_results = evaluate_dataset(ds_name, gt_root, args.pred_root, args.models)
        
        if not dataset_results:
            continue
        
        # 绘制 Success Plot
        plot_success_curves(dataset_results, args.output_dir, ds_name)
        
        # 计算标量指标并填表
        for model in args.models:
            ious = np.array(dataset_results[model])
            if len(ious) == 0:
                summary_table.add_row([ds_name, model, 0.0, 0.0, 0.0])
                continue
                
            # AUC: 0-1 阈值下的平均成功率
            thresholds = np.linspace(0, 1, 21)
            curve = [np.mean(ious >= thr) for thr in thresholds]
            auc = np.mean(curve)
            
            # OP@0.5: IoU >= 0.5 的比例
            op50 = np.mean(ious >= 0.50)
            
            # OP@0.75: IoU >= 0.75 的比例
            op75 = np.mean(ious >= 0.75)
            
            summary_table.add_row([ds_name, model, f"{auc:.3f}", f"{op50:.3f}", f"{op75:.3f}"])
            
        summary_table.add_row(["---", "---", "---", "---", "---"])

    print("\n" + "="*60)
    print("评测结果汇总")
    print("="*60)
    print(summary_table)
    
    # 保存表格
    report_path = os.path.join(args.output_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(str(summary_table))
    print(f"\n📊 报告已保存: {report_path}")


if __name__ == "__main__":
    main()

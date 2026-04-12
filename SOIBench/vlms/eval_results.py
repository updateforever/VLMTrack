# -*- coding: utf-8 -*-
"""
SOIBench/vlms/eval_results.py
评测 Grounding 结果
功能：
1）加载 GT 和 Pred JSONL
2）计算 IoU 指标
3）绘制 Success Plot
4）生成评测报告（整体 + 各子数据集）
"""

import argparse
import glob
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from tqdm import tqdm


def _pick_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _resolve_default_gt_roots():
    lasot = _pick_existing([
        os.environ.get('SOIBENCH_LASOT_JSONL'),
        '/root/user-data/wyp/VLMTrack/SOIBench/data/test/lasot',
        '/home/member/data2/wyp/SUTrack/SOIBench/data/test/lasot',
    ]) or ''
    mgit = _pick_existing([
        os.environ.get('SOIBENCH_MGIT_JSONL'),
        '/root/user-data/wyp/VLMTrack/SOIBench/data/test/mgit',
        '/home/member/data2/wyp/SUTrack/SOIBench/data/test/mgit',
    ]) or ''
    tnl2k = _pick_existing([
        os.environ.get('SOIBENCH_TNL2K_JSONL'),
        '/root/user-data/wyp/VLMTrack/SOIBench/data/test/tnl2k',
        '/home/member/data2/wyp/SUTrack/SOIBench/data/test/tnl2k',
    ]) or ''
    return lasot, mgit, tnl2k


# 默认模型名映射表（存储名 -> 显示名）
DEFAULT_MODEL_NAME_MAP = {
    # Qwen3VL 系列
    "local_qwen3-vl-4b-instruct": "Qwen3-VL-4B",
    "local_qwen3-vl-8b-instruct": "Qwen3-VL-8B",
    "api_qwen3-vl-235b-a22b-instruct": "Qwen3-VL-235B",
    "qwen3vl_4b": "Qwen3-VL-4B",
    "qwen3vl_api": "Qwen3-VL-API",
    
    # 其他常见模型
    "gpt4v": "GPT-4V",
    "gemini": "Gemini-Pro",
    "claude": "Claude-3",
    
    # 人类基线
    "Human_Baseline": "Human",
}


def get_display_name(model_tag, name_map=None):
    """
    获取模型的显示名称
    参数:
        model_tag: 模型存储标签
        name_map: 自定义名称映射字典
    返回:
        显示名称
    """
    if name_map and model_tag in name_map:
        return name_map[model_tag]
    if model_tag in DEFAULT_MODEL_NAME_MAP:
        return DEFAULT_MODEL_NAME_MAP[model_tag]
    return model_tag  # 如果没有映射，返回原始名称


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


def load_seq_data(jsonl_path, is_gt=False, load_human_baseline=False, skip_consecutive_frames=False):
    """
    加载单个序列文件，返回 {frame_idx: box} 字典
    参数:
        jsonl_path: JSONL 文件路径
        is_gt: 是否为 GT 文件
        load_human_baseline: 是否加载人类基线 (从 GT 文件的 pred_boxes 字段)
        skip_consecutive_frames: 是否跳过连续帧 (True=严格模式, False=完整模式)
    返回:
        {frame_idx: box} 字典
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

                if load_human_baseline:
                    # 加载人类基线: 从 pred_boxes 提取
                    pred_boxes = item.get("pred_boxes", [])
                    
                    if is_skip:
                        # skip 帧: 使用上一个有效帧的结果
                        if last_valid_box is not None:
                            data_map[fid] = last_valid_box
                    else:
                        # 非 skip 帧: 提取 pred_boxes
                        if pred_boxes and len(pred_boxes) > 0:
                            # pred_boxes 格式: [[x1,y1], [x2,y2]] -> [x1,y1,x2,y2]
                            box = pred_boxes
                            if len(box) == 2 and len(box[0]) == 2:
                                box = [box[0][0], box[0][1], box[1][0], box[1][1]]
                            last_valid_box = box
                            data_map[fid] = box
                        
                elif is_gt:
                    # GT 提取逻辑
                    # 注意: 如果 skip_consecutive_frames=True，则跳过 skip 帧
                    # 如果 skip_consecutive_frames=False，则加载所有帧（算法评测所有帧）
                    if skip_consecutive_frames and is_skip:
                        continue  # 严格模式: 跳过 skip 帧
                    
                    box = item.get("gt_box") or item.get("bbox")
                    # gt_box 格式: [[x1,y1], [x2,y2]] -> [x1,y1,x2,y2]
                    if box and len(box) == 2 and len(box[0]) == 2:
                        box = [box[0][0], box[0][1], box[1][0], box[1][1]]
                    if box:
                        data_map[fid] = box
                else:
                    # Pred 提取逻辑: 取第一个预测框
                    p_boxes = item.get("parsed_bboxes") or item.get("parsed_bbox")
                    box = p_boxes[0] if (p_boxes and len(p_boxes) > 0) else None
                    if box:
                        data_map[fid] = box
                        
            except:
                continue
    return data_map


def evaluate_dataset(ds_name, gt_root, pred_root, model_tags, add_human_baseline=False, skip_consecutive_frames=False):
    """
    评测单个数据集
    参数:
        ds_name: 数据集名称
        gt_root: GT JSONL 文件根目录
        pred_root: 预测结果根目录
        model_tags: 模型标签列表
        add_human_baseline: 是否添加人类基线
        skip_consecutive_frames: 是否跳过连续帧
            - True: 严格模式，算法和人类都只评测非 skip 帧
            - False: 完整模式，算法评测所有帧，人类在 skip 帧复用上一帧
    返回:
        {model_name: [all_ious_list]} 字典
    """
    # 查找该数据集所有 GT 文件
    gt_files = sorted(glob.glob(os.path.join(gt_root, "*.jsonl")))
    if not gt_files:
        print(f"⚠️  {ds_name} 未找到 GT 文件，跳过。")
        return {}

    # 存储每个模型在该数据集下的所有帧 IoU
    all_tags = model_tags + (["Human_Baseline"] if add_human_baseline else [])
    model_ious = {tag: [] for tag in all_tags}
    
    print(f"🔄 正在评测 {ds_name} ({len(gt_files)} 个序列)...")

    for gt_path in tqdm(gt_files, leave=False, desc=f"评测 {ds_name}"):
        seq_name = os.path.basename(gt_path).replace("_descriptions.jsonl", "").replace(".jsonl", "")
        
        # 加载 GT {frame: box}
        gt_map = load_seq_data(gt_path, is_gt=True, skip_consecutive_frames=skip_consecutive_frames)
        if not gt_map:
            continue
        
        # 加载人类基线 (如果需要)
        human_map = {}
        if add_human_baseline:
            human_map = load_seq_data(gt_path, is_gt=False, load_human_baseline=True)

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
        
        # 计算人类基线 IoU
        if add_human_baseline:
            for fid, gt_box in gt_map.items():
                human_box = human_map.get(fid)
                iou = calculate_iou(human_box, gt_box)
                model_ious["Human_Baseline"].append(iou)
                
    return model_ious


def plot_success_curves(results, output_dir, ds_name, name_map=None):
    """
    绘制 Success Plot (成功率曲线)
    参数:
        results: {model_name: [iou_list]} 字典
        output_dir: 输出目录
        ds_name: 数据集名称
        name_map: 模型名称映射字典
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
        display_name = get_display_name(model_name, name_map)
        plt.plot(thresholds, curve, linewidth=2, label=f"{display_name} [{auc:.3f}]")
        
    plt.legend(loc='lower left', fontsize=12)
    
    save_path = os.path.join(output_dir, f"{ds_name}_success_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 曲线图已保存: {save_path}")


def main():
    default_lasot_gt, default_mgit_gt, default_tnl2k_gt = _resolve_default_gt_roots()

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
                        default=default_lasot_gt,
                        help="LaSOT GT JSONL 根目录")
    parser.add_argument("--mgit_gt_root", type=str, 
                        default=default_mgit_gt,
                        help="MGIT GT JSONL 根目录")
    parser.add_argument("--tnl2k_gt_root", type=str, 
                        default=default_tnl2k_gt,
                        help="TNL2K GT JSONL 根目录")
    
    # 人类基线
    parser.add_argument("--add_human_baseline", action="store_true",
                        help="是否添加人类基线对比 (从 GT JSONL 的 pred_boxes 字段提取)")
    parser.add_argument("--skip_consecutive_frames", action="store_true",
                        help="是否跳过连续帧评测 (严格模式: 算法和人类都只评测 SOI 帧; 默认: 算法评测所有帧，人类复用)")
    parser.add_argument("--model_names", type=str, default=None,
                        help="模型名称映射，格式: 'tag1:Name1,tag2:Name2'，例如: 'local_qwen3-vl-4b:Qwen-4B,api_model:API-Model'")

    args = parser.parse_args()
    
    # 解析自定义模型名映射
    custom_name_map = {}
    if args.model_names:
        for pair in args.model_names.split(','):
            if ':' in pair:
                tag, name = pair.split(':', 1)
                custom_name_map[tag.strip()] = name.strip()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 构建 GT 根目录字典
    gt_roots = {
        "lasot": args.lasot_gt_root,
        "mgit": args.mgit_gt_root,
        "tnl2k": args.tnl2k_gt_root
    }
    
    print("\n" + "="*60)
    print("SOIBench Grounding 评测")
    print("="*60)
    
    # 收集所有数据集的结果
    all_models = args.models + (["Human_Baseline"] if args.add_human_baseline else [])
    all_dataset_results = {model: [] for model in all_models}  # {model: [所有数据集的 IoU 合并]}
    per_dataset_results = {}  # {dataset: {model: [IoU]}}
    
    for ds_name in args.datasets:
        gt_root = gt_roots.get(ds_name)
        
        if not gt_root or not os.path.exists(gt_root):
            print(f"❌ 无法找到 {ds_name} 的 GT 目录: {gt_root}，跳过")
            continue
        
        # 计算该数据集下所有模型的 IoU
        dataset_results = evaluate_dataset(ds_name, gt_root, args.pred_root, args.models, args.add_human_baseline, args.skip_consecutive_frames)
        
        if not dataset_results:
            continue
        
        # 保存该数据集的结果
        per_dataset_results[ds_name] = dataset_results
        
        # 合并到总结果中
        for model in all_models:
            all_dataset_results[model].extend(dataset_results[model])
        
        # 绘制该数据集的 Success Plot
        plot_success_curves(dataset_results, args.output_dir, ds_name, custom_name_map)
    
    # 绘制整体 SOIBench Success Plot
    if any(all_dataset_results.values()):
        plot_success_curves(all_dataset_results, args.output_dir, "SOIBench_Overall", custom_name_map)
    
    # 生成报告表格
    print("\n" + "="*60)
    print("评测结果汇总")
    print("="*60)
    
    # 1. 整体 SOIBench 指标
    if any(all_dataset_results.values()):
        print("\n【整体 SOIBench 指标】")
        overall_table = PrettyTable()
        overall_table.field_names = ["Model", "AUC", "OP@0.50", "OP@0.75", "Total Frames"]
        
        for model in all_models:
            ious = np.array(all_dataset_results[model])
            if len(ious) == 0:
                overall_table.add_row([model, 0.0, 0.0, 0.0, 0])
                continue
                
            # AUC: 0-1 阈值下的平均成功率
            thresholds = np.linspace(0, 1, 21)
            curve = [np.mean(ious >= thr) for thr in thresholds]
            auc = np.mean(curve)
            
            # OP@0.5: IoU >= 0.5 的比例
            op50 = np.mean(ious >= 0.50)
            
            # OP@0.75: IoU >= 0.75 的比例
            op75 = np.mean(ious >= 0.75)
            
            overall_table.add_row([get_display_name(model, custom_name_map), f"{auc:.3f}", f"{op50:.3f}", f"{op75:.3f}", len(ious)])
        
        print(overall_table)
    
    # 2. 各子数据集指标
    if per_dataset_results:
        print("\n【各子数据集指标】")
        detail_table = PrettyTable()
        detail_table.field_names = ["Dataset", "Model", "AUC", "OP@0.50", "OP@0.75", "Frames"]
        
        for ds_name in args.datasets:
            if ds_name not in per_dataset_results:
                continue
                
            dataset_results = per_dataset_results[ds_name]
            
            for model in all_models:
                ious = np.array(dataset_results[model])
                if len(ious) == 0:
                    detail_table.add_row([ds_name, model, 0.0, 0.0, 0.0, 0])
                    continue
                    
                # AUC: 0-1 阈值下的平均成功率
                thresholds = np.linspace(0, 1, 21)
                curve = [np.mean(ious >= thr) for thr in thresholds]
                auc = np.mean(curve)
                
                # OP@0.5: IoU >= 0.5 的比例
                op50 = np.mean(ious >= 0.50)
                
                # OP@0.75: IoU >= 0.75 的比例
                op75 = np.mean(ious >= 0.75)
                
                detail_table.add_row([ds_name, get_display_name(model, custom_name_map), f"{auc:.3f}", f"{op50:.3f}", f"{op75:.3f}", len(ious)])
            
            detail_table.add_row(["---", "---", "---", "---", "---", "---"])
        
        print(detail_table)
    
    # 保存报告
    report_path = os.path.join(args.output_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("SOIBench Grounding 评测报告\n")
        f.write("="*60 + "\n\n")
        
        if any(all_dataset_results.values()):
            f.write("【整体 SOIBench 指标】\n")
            f.write(str(overall_table) + "\n\n")
        
        if per_dataset_results:
            f.write("【各子数据集指标】\n")
            f.write(str(detail_table) + "\n")
    
    print(f"\n📊 报告已保存: {report_path}")


if __name__ == "__main__":
    main()

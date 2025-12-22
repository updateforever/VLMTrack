# -*- coding: utf-8 -*-
import argparse
import glob
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from tqdm import tqdm

# =========================================================
# 1. 配置区域
# =========================================================
# 原始标注文件的根目录 (Source GT)
DATASET_GT_ROOTS = {
    "lasot": "/home/member/data1/DATASETS_PUBLIC/LaSOT/LaSOTBenchmark",
    # 如果您的 jsonl 在其他位置，请修改这里。例如之前提到的：
    # "lasot": "/home/member/data2/wyp/SUTrack/SOIBench/data/test/lasot",
    "mgit":  "/home/member/data2/wyp/SUTrack/SOIBench/data/test/mgit",
    "tnl2k": "/home/member/data2/wyp/SUTrack/SOIBench/data/test/tnl2k"
}

# =========================================================
# 2. 核心计算函数
# =========================================================
def calculate_iou(box1, box2):
    """
    计算 IoU
    box: [x1, y1, x2, y2]
    """
    if not box1 or not box2: return 0.0
    
    # 兼容可能出现的嵌套 list
    b1 = [float(x) for x in (box1[0] if isinstance(box1[0], list) else box1)]
    b2 = [float(x) for x in (box2[0] if isinstance(box2[0], list) else box2)]
    
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    
    area1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    area2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = area1 + area2 - inter_area
    
    return inter_area / union if union > 0 else 0.0

def load_seq_data(jsonl_path, is_gt=False):
    """
    加载单个序列文件，返回 {frame_idx: box} 字典
    """
    data_map = {}
    if not os.path.exists(jsonl_path):
        return data_map

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                item = json.loads(line)
                # 跳过 skip 的帧 (仅针对 GT)
                if is_gt and item.get("status") == "skip":
                    continue
                
                fid = int(item.get("frame_idx", -1))
                if fid == -1: continue

                if is_gt:
                    # GT 提取逻辑
                    box = item.get("gt_box") or item.get("bbox")
                else:
                    # Pred 提取逻辑 (约定输出字段为 parsed_bboxes)
                    # 我们取第一个框作为预测结果
                    p_boxes = item.get("parsed_bboxes") or item.get("parsed_bbox")
                    box = p_boxes[0] if (p_boxes and len(p_boxes) > 0) else None
                
                if box:
                    data_map[fid] = box
            except:
                continue
    return data_map

# =========================================================
# 3. 评测主逻辑
# =========================================================
def evaluate_dataset(ds_name, gt_root, pred_root, model_tags):
    """
    评测单个数据集
    返回: {model_name: [all_ious_list]}
    """
    # 1. 找到该数据集所有 GT 文件
    gt_files = sorted(glob.glob(os.path.join(gt_root, "*.jsonl")))
    if not gt_files:
        print(f"⚠️  {ds_name} 未找到 GT 文件，跳过。")
        return {}

    # 存储每个模型在该数据集下的所有帧 IoU
    model_ious = {tag: [] for tag in model_tags}
    
    print(f"🔄 正在评测 {ds_name} ({len(gt_files)} 序列)...")

    for gt_path in tqdm(gt_files, leave=False):
        seq_name = os.path.basename(gt_path).replace("_descriptions.jsonl", "").replace(".jsonl", "")
        
        # A. 加载 GT {frame: box}
        gt_map = load_seq_data(gt_path, is_gt=True)
        if not gt_map: continue

        # B. 遍历每个模型
        for tag in model_tags:
            # 构造预测文件名: {seq_name}_{tag}_pred.jsonl
            # 假设所有模型结果都在 pred_root/dataset_name/ 下
            pred_filename = f"{seq_name}_{tag}_pred.jsonl"
            pred_path = os.path.join(pred_root, ds_name, pred_filename)
            
            # 加载 Pred {frame: box}
            pred_map = load_seq_data(pred_path, is_gt=False)
            
            # C. 逐帧对齐计算 (以 GT 为准)
            for fid, gt_box in gt_map.items():
                pred_box = pred_map.get(fid)
                iou = calculate_iou(pred_box, gt_box)
                model_ious[tag].append(iou)
                
    return model_ious

# =========================================================
# 4. 绘图与报表
# =========================================================
def plot_success_curves(results, output_dir, ds_name):
    """
    绘制 Success Plot
    results: {model_name: [iou_list]}
    """
    plt.figure(figsize=(10, 7))
    plt.title(f"Success Plot - {ds_name.upper()}")
    plt.xlabel("Overlap Threshold (IoU)")
    plt.ylabel("Success Rate")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    thresholds = np.linspace(0, 1, 21) # 0, 0.05, ..., 1.0
    
    # 按 AUC 排序图例
    model_stats = []
    
    for model_name, ious in results.items():
        if not ious: continue
        ious_arr = np.array(ious)
        
        # 计算曲线点
        curve = [np.mean(ious_arr >= thr) for thr in thresholds]
        auc = np.mean(curve) # 近似 AUC
        model_stats.append((model_name, auc, curve))
        
    # 排序
    model_stats.sort(key=lambda x: x[1], reverse=True)
    
    for model_name, auc, curve in model_stats:
        plt.plot(thresholds, curve, linewidth=2, label=f"{model_name} [{auc:.3f}]")
        
    plt.legend(loc='lower left')
    
    save_path = os.path.join(output_dir, f"{ds_name}_success_plot.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📈 曲线图已保存: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_root", type=str, default="./results", help="预测结果根目录")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="评测结果保存目录")
    parser.add_argument("--models", nargs='+', required=True, help="要对比的模型 tag 列表，例如: qwen3vl_v1 internvl2_v1")
    parser.add_argument("--datasets", nargs='+', default=["lasot", "mgit", "tnl2k"], help="要评测的数据集")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 总表
    summary_table = PrettyTable()
    summary_table.field_names = ["Dataset", "Model", "AUC", "OP@0.50", "OP@0.75"]
    
    for ds_name in args.datasets:
        # 获取 GT 根目录
        # 这里为了演示，假设 args.datasets 里的名字能对应到 DATASET_GT_ROOTS 的 key
        # 如果您的 jsonl 都在一个统一路径下，可以手动硬编码
        gt_root = DATASET_GT_ROOTS.get(ds_name)
        
        # 这里做一个 fallback，防止 key 不匹配
        if not gt_root:
            # 尝试根据之前的路径习惯猜测
            possible_path = f"/home/member/data2/wyp/SUTrack/SOIBench/data/test/{ds_name}"
            if os.path.exists(possible_path):
                gt_root = possible_path
            else:
                print(f"❌ 无法找到 {ds_name} 的 GT 目录，跳过")
                continue
        
        # 1. 计算该数据集下所有模型的 IoU
        # 结构: {model_tag: [0.9, 0.8, 0.0, ...]}
        dataset_results = evaluate_dataset(ds_name, gt_root, args.pred_root, args.models)
        
        if not dataset_results: continue
        
        # 2. 绘图
        plot_success_curves(dataset_results, args.output_dir, ds_name)
        
        # 3. 计算标量指标并填表
        for model in args.models:
            ious = np.array(dataset_results[model])
            if len(ious) == 0:
                summary_table.add_row([ds_name, model, 0.0, 0.0, 0.0])
                continue
                
            # AUC (简单计算为 0-1 阈值下的平均成功率)
            thresholds = np.linspace(0, 1, 21)
            curve = [np.mean(ious >= thr) for thr in thresholds]
            auc = np.mean(curve)
            
            # OP@0.5 (Precision at IoU=0.5)
            op50 = np.mean(ious >= 0.50)
            
            # OP@0.75
            op75 = np.mean(ious >= 0.75)
            
            summary_table.add_row([ds_name, model, f"{auc:.3f}", f"{op50:.3f}", f"{op75:.3f}"])
            
        summary_table.add_row(["---", "---", "---", "---", "---"])

    print("\n" + str(summary_table))
    
    # 保存表格
    with open(os.path.join(args.output_dir, "report.txt"), "w") as f:
        f.write(str(summary_table))

if __name__ == "__main__":
    main()
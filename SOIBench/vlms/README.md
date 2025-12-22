# SOIBench Grounding 评测工具

## 简介

这是一套用于评测 VLM (Vision Language Model) 在 SOI (Semantic Object Identification) 文本引导帧 Grounding 检测任务上的工具集。

## 功能

1. **run_grounding_qwen3vl.py**: Grounding 推理主脚本
   - 支持本地模型和 API 调用
   - 自动解析多种 bbox 格式
   - 支持断点续跑
   - 可选保存可视化结果

2. **eval_results.py**: 评测脚本
   - 计算 IoU 指标
   - 绘制 Success Plot
   - 生成评测报告

3. **visualize_grounding.py**: 可视化脚本
   - 在原图上绘制 GT 和 Pred bbox
   - 支持保存为图片或视频

## 安装依赖

```bash
pip install torch transformers pillow opencv-python numpy matplotlib prettytable tqdm
```

## 使用方法

### 1. 运行 Grounding 推理

#### 使用本地模型

```bash
python run_grounding_qwen3vl.py \
    --mode local \
    --model_path /path/to/Qwen2-VL-7B-Instruct \
    --lasot_jsonl /path/to/lasot_jsonl \
    --lasot_root /path/to/lasot_images \
    --output_root ./results \
    --exp_tag v1
```

#### 使用 API

```bash
python run_grounding_qwen3vl.py \
    --mode api \
    --api_model_name qwen-vl-max \
    --api_key_env YOUR_API_KEY \
    --lasot_jsonl /path/to/lasot_jsonl \
    --lasot_root /path/to/lasot_images \
    --output_root ./results \
    --exp_tag api_v1
```

**主要参数说明:**

- `--mode`: 推理模式，`local` (本地模型) 或 `api` (API 调用)
- `--model_path`: 本地模型路径 (mode=local 时必需)
- `--lasot_root/mgit_root/tnl2k_root`: 数据集图像根目录
- `--lasot_jsonl/mgit_jsonl/tnl2k_jsonl`: JSONL 描述文件目录
- `--output_root`: 结果保存根目录
- `--exp_tag`: 实验标签
- `--save_debug_vis`: 是否保存调试可视化

### 2. 评测结果

```bash
python eval_results.py \
    --pred_root ./results \
    --output_dir ./eval_results \
    --models local_v1 api_v1 \
    --datasets lasot mgit tnl2k \
    --lasot_gt_root /path/to/lasot_jsonl \
    --mgit_gt_root /path/to/mgit_jsonl \
    --tnl2k_gt_root /path/to/tnl2k_jsonl
```

**主要参数说明:**

- `--pred_root`: 预测结果根目录
- `--output_dir`: 评测结果保存目录
- `--models`: 要对比的模型标签列表
- `--datasets`: 要评测的数据集
- `--lasot_gt_root/mgit_gt_root/tnl2k_gt_root`: GT JSONL 文件根目录

**输出:**

- `{dataset}_success_plot.png`: Success Plot 曲线图
- `report.txt`: 评测报告表格

**评测指标:**

- **AUC**: Success Plot 曲线下面积 (0-1 阈值平均成功率)
- **OP@0.50**: IoU >= 0.5 的比例
- **OP@0.75**: IoU >= 0.75 的比例

### 3. 可视化结果

#### 保存为图片

```bash
python visualize_grounding.py \
    --dataset lasot \
    --seq_name airplane-1 \
    --pred_file ./results/lasot/local_v1/airplane-1_pred.jsonl \
    --gt_file /path/to/lasot_jsonl/airplane-1_descriptions.jsonl \
    --image_root /path/to/lasot_images \
    --output_dir ./vis_results
```

#### 保存为视频

```bash
python visualize_grounding.py \
    --dataset lasot \
    --seq_name airplane-1 \
    --pred_file ./results/lasot/local_v1/airplane-1_pred.jsonl \
    --gt_file /path/to/lasot_jsonl/airplane-1_descriptions.jsonl \
    --image_root /path/to/lasot_images \
    --output_dir ./vis_results \
    --save_video \
    --fps 30
```

**主要参数说明:**

- `--dataset`: 数据集名称 (lasot/mgit/tnl2k)
- `--seq_name`: 序列名称
- `--pred_file`: 预测结果 JSONL 文件路径
- `--gt_file`: GT JSONL 文件路径
- `--image_root`: 图片根目录
- `--save_video`: 是否保存为视频
- `--fps`: 视频帧率

## 目录结构

```
SOIBench/vlms/
├── run_grounding_qwen3vl.py    # Grounding 推理主脚本
├── eval_results.py              # 评测脚本
├── visualize_grounding.py       # 可视化脚本
├── qwen3vl_infer.py            # Qwen3VL 推理引擎
├── run_eval.sh                  # 示例运行脚本
└── README.md                    # 本文档
```

## 输出文件格式

### 预测结果 JSONL

每行一个 JSON 对象，包含:

```json
{
  "frame_idx": 0,
  "image_path": "...",
  "gt_box": [x1, y1, x2, y2],
  "model_raw_response": "...",
  "parsed_bboxes": [[x1, y1, x2, y2]],
  "parse_status": "ok"
}
```

### 评测报告

```
+----------+----------+-------+---------+---------+
| Dataset  | Model    | AUC   | OP@0.50 | OP@0.75 |
+----------+----------+-------+---------+---------+
| lasot    | local_v1 | 0.723 | 0.856   | 0.612   |
| mgit     | local_v1 | 0.698 | 0.834   | 0.587   |
| tnl2k    | local_v1 | 0.701 | 0.841   | 0.593   |
+----------+----------+-------+---------+---------+
```

## 注意事项

1. **路径配置**: 请根据实际情况修改数据集路径
2. **断点续跑**: 脚本支持断点续跑，重复运行会跳过已处理的帧
3. **内存占用**: 本地模型推理需要较大显存，建议使用 GPU
4. **API 限流**: 使用 API 时注意速率限制

## 常见问题

**Q: 如何添加新的数据集?**

A: 在脚本中添加对应的 `--{dataset}_root` 和 `--{dataset}_jsonl` 参数即可。

**Q: 如何修改 bbox 解析逻辑?**

A: 修改 `run_grounding_qwen3vl.py` 中的 `extract_bboxes_from_model_output` 函数。

**Q: 如何自定义评测指标?**

A: 修改 `eval_results.py` 中的 `evaluate_dataset` 和 `plot_success_curves` 函数。

## 更新日志

- **2025-12-22**: 初始版本
  - 支持 Qwen3VL 本地和 API 推理
  - 实现 IoU 评测和 Success Plot
  - 添加可视化工具

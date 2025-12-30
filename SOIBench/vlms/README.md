# SOIBench Grounding 评测工具

## 简介

这是一套用于评测 VLM (Vision Language Model) 在 SOI (Semantic Object Identification) 文本引导帧 Grounding 检测任务上的工具集。

**✨ 新特性**: 采用**适配器模式**架构，支持任意 VLM 模型的快速接入！

## 🚀 快速开始

### 使用现有模型

```bash
# Qwen3VL API 推理
python run_grounding.py --model qwen3vl --mode api

# GLM-4.6V 本地推理
python run_grounding.py --model glm46v --mode local

# DeepSeek-VL2 API 推理 + 可视化
python run_grounding.py --model deepseekvl --mode api --save_debug_vis
```

### 添加新模型

只需3步即可接入新的 VLM 模型！详见 [GROUNDING_FRAMEWORK.md](GROUNDING_FRAMEWORK.md)

## 架构说明

```
SOIBench/vlms/
├── model_adapters/              # 📦 模型适配器
│   ├── base.py                 # 抽象基类
│   ├── qwen3vl_adapter.py      # Qwen3VL
│   ├── glm46v_adapter.py       # GLM-4.6V
│   └── deepseekvl_adapter.py   # DeepSeek-VL2
├── grounding_common.py          # 🔧 通用函数和主流程
├── run_grounding.py             # 🚀 统一入口脚本
├── eval_results.py              # 📊 评测脚本
├── visualize_grounding.py       # 🎨 可视化脚本
├── qwen3vl_infer.py            # Qwen3VL 推理引擎
├── glm46v_infer.py             # GLM-4.6V 推理引擎
├── deepseekvl_infer.py         # DeepSeek-VL2 推理引擎
└── legacy/                      # 旧版脚本（已废弃）
```

## 功能

### 1. 统一推理入口 (`run_grounding.py`)

支持多种 VLM 模型的 Grounding 推理：

```bash
python run_grounding.py \
  --model {qwen3vl|glm46v|deepseekvl} \
  --mode {local|api} \
  --output_root ./results \
  --exp_tag my_experiment \
  --save_debug_vis
```

**主要参数:**
- `--model`: 模型名称 (qwen3vl, glm46v, deepseekvl)
- `--mode`: 推理模式 (local 本地模型, api API调用)
- `--model_path`: 本地模型路径 (mode=local 时)
- `--api_key`: API 密钥 (mode=api 时，或从环境变量读取)
- `--output_root`: 结果保存根目录
- `--exp_tag`: 实验标签
- `--save_debug_vis`: 是否保存可视化

**特性:**
- ✅ 自动路径修复
- ✅ 断点续跑
- ✅ 多种 bbox 格式解析
- ✅ 可选可视化

### 2. 评测脚本 (`eval_results.py`)

计算 IoU 指标并生成评测报告：

```bash
python eval_results.py \
  --pred_root ./results \
  --output_dir ./eval_results \
  --models qwen3vl glm46v deepseekvl \
  --datasets lasot mgit tnl2k \
  --lasot_gt_root /path/to/lasot_jsonl \
  --mgit_gt_root /path/to/mgit_jsonl \
  --tnl2k_gt_root /path/to/tnl2k_jsonl
```

**评测指标:**
- **AUC**: Success Plot 曲线下面积
- **OP@0.50**: IoU >= 0.5 的比例
- **OP@0.75**: IoU >= 0.75 的比例

**输出:**
- Success Plot 曲线图
- 评测报告表格

### 3. 可视化脚本 (`visualize_grounding.py`)

在原图上绘制 GT 和预测 bbox：

```bash
# 保存为图片序列
python visualize_grounding.py \
  --dataset lasot \
  --seq_name airplane-1 \
  --pred_file ./results/lasot/api_qwen3vl/airplane-1_pred.jsonl \
  --gt_file /path/to/lasot_jsonl/airplane-1_descriptions.jsonl \
  --image_root /path/to/lasot_images \
  --output_dir ./vis_results

# 保存为视频
python visualize_grounding.py \
  --dataset lasot \
  --seq_name airplane-1 \
  --pred_file ./results/lasot/api_qwen3vl/airplane-1_pred.jsonl \
  --gt_file /path/to/lasot_jsonl/airplane-1_descriptions.jsonl \
  --image_root /path/to/lasot_images \
  --output_dir ./vis_results \
  --save_video \
  --fps 30
```

## 安装依赖

```bash
pip install torch transformers pillow opencv-python numpy matplotlib prettytable tqdm dashscope openai
```

## 支持的模型

| 模型 | 本地推理 | API 推理 | 默认 API |
|------|---------|---------|---------|
| Qwen3VL | ✅ | ✅ | DashScope |
| GLM-4.6V | ✅ | ✅ | SiliconFlow |
| DeepSeek-VL2 | ✅ | ✅ | SiliconFlow |

## 环境变量

```bash
# Qwen3VL
export DASHSCOPE_API_KEY='your-key'

# GLM-4.6V / DeepSeek-VL2
export SILICONFLOW_API_KEY='your-key'
```

## 输出格式

所有模型的输出格式完全一致：

```json
{
  "frame_idx": 0,
  "image_path": "...",
  "gt_box": [[x1, y1], [x2, y2]],
  "output-en": {...},
  "model_response": "...",
  "parsed_bboxes": [[x1, y1, x2, y2], ...]
}
```

## 完整工作流示例

```bash
# 1. 推理
python run_grounding.py --model qwen3vl --mode api --exp_tag exp1

# 2. 评测
python eval_results.py \
  --pred_root ./results \
  --models api_exp1 \
  --datasets lasot mgit tnl2k

# 3. 可视化
python visualize_grounding.py \
  --dataset lasot \
  --seq_name airplane-1 \
  --pred_file ./results/lasot/api_exp1/airplane-1_pred.jsonl
```

## 扩展新模型

详细教程请参考: [GROUNDING_FRAMEWORK.md](GROUNDING_FRAMEWORK.md)

简要步骤：
1. 创建推理引擎 (`myvlm_infer.py`)
2. 创建适配器 (`model_adapters/myvlm_adapter.py`)
3. 注册适配器 (`model_adapters/__init__.py`)

然后就可以使用：
```bash
python run_grounding.py --model myvlm --mode api
```

## 注意事项

1. **路径配置**: 请根据实际情况修改数据集路径
2. **断点续跑**: 脚本支持断点续跑，重复运行会跳过已处理的帧
3. **内存占用**: 本地模型推理需要较大显存
4. **API 限流**: 使用 API 时注意速率限制

## 常见问题

**Q: 旧的 `run_grounding_qwen3vl.py` 等脚本还能用吗?**

A: 可以，但已移至 `legacy/` 目录，不再维护。建议使用新的 `run_grounding.py`。

**Q: 如何添加新的数据集?**

A: 在 `run_grounding.py` 中添加对应的参数即可，主流程会自动处理。

**Q: 如何自定义评测指标?**

A: 修改 `eval_results.py` 中的 `evaluate_dataset` 函数。

## 更新日志

- **2025-12-30**: 重构为适配器架构
  - ✅ 统一推理入口
  - ✅ 支持 Qwen3VL, GLM-4.6V, DeepSeek-VL2
  - ✅ 易于扩展新模型
  
- **2025-12-22**: 初始版本
  - 支持 Qwen3VL 推理
  - 实现 IoU 评测
  - 添加可视化工具

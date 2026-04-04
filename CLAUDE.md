# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指导。

## 项目概述

VLMTrack 旨在实现**认知跟踪（Cognitive Tracking）**，突破传统感知跟踪的局限。

### 核心理念

**传统感知跟踪的局限**：
- 纯视觉特征匹配，缺乏语义理解
- 局部搜索窗口限制（依赖前一帧位置）
- 无法推理目标状态（遮挡、消失、重现）
- 必须每帧输出预测框，即使目标不可见

**VLT（Vision-Language Tracking）的不足**：
- 仅在初始帧使用文本描述目标
- 后续仍依赖视觉特征匹配
- 本质仍是感知跟踪

**我们的认知跟踪**：
- **全图搜索**：不受局部窗口限制，VLM 在整张图中推理目标位置
- **文本输入输出**：
  - 输入：当前帧 + 文本描述（目标特征、历史状态）
  - 输出：文本描述跟踪状态（"目标在左上角"、"目标被遮挡"、"目标不在视野"）+ bbox（如可见）
- **自启发认知推理**：理解场景语义，维持对目标的精准感知和查询

### 技术基础

项目基于 SUTrack（AAAI2025）统一框架，整合五种 SOT 任务：RGB、RGB-Depth、RGB-Thermal、RGB-Event、RGB-Language。通过 Qwen3-VL 等视觉语言模型实现认知跟踪能力。

## 环境配置

```bash
# 创建环境
conda create -n sutrack python=3.8
conda activate sutrack
bash install.sh

# 设置 Python 路径（必须）
export PYTHONPATH=/data/wyp/VLMTrack:$PYTHONPATH

# 配置路径
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .

# VLM API 模式需要设置
export DASHSCOPE_API_KEY=<your_key>
```

## 常用命令

### 训练（SUTrack）
```bash
# 多卡训练（4卡）
python -m torch.distributed.launch --nproc_per_node 4 lib/train/run_training.py --script sutrack --config sutrack_b224 --save_dir .

# 单卡调试
python tracking/train.py --script sutrack --config sutrack_b224 --save_dir . --mode single
```

### RGB 跟踪测试
```bash
# LaSOT
python tracking/test.py sutrack sutrack_b224 --dataset lasot --threads 2
python tracking/analysis_results.py

# GOT-10K（需要特殊格式转换）
python tracking/test.py sutrack sutrack_b224 --dataset got10k_test --threads 2
python lib/test/utils/transform_got10k.py --tracker_name sutrack --cfg_name sutrack_b224

# TrackingNet
python tracking/test.py sutrack sutrack_b224 --dataset trackingnet --threads 2
```

### VLM 跟踪器测试
```bash
# 认知跟踪器（核心，使用结构化状态输出）
python tracking/test.py qwen_vlm_cognitive default --dataset lasot --debug 1
python tracking/test.py qwen_vlm_cognitive default --dataset videocube_val --debug 1

# 纯视觉跟踪器（2帧或3帧模式）
python tracking/test.py qwen_vlm_visual default_2f --dataset lasot --debug 1
python tracking/test.py qwen_vlm_visual default_3f --dataset lasot --debug 1

# 混合跟踪器（视觉+认知混合）
python tracking/test.py qwen_vlm_hybrid default_visual_kf --dataset lasot --debug 1
python tracking/test.py qwen_vlm_hybrid default_cognitive_conf --dataset lasot --debug 1

# 使用测试脚本（认知跟踪）
bash test_cognitive_tracking.sh
```

### 多模态测试
```bash
# RGB-Thermal（LasHeR）
python ./RGBT_workspace/test_rgbt_mgpus.py --script_name sutrack --dataset_name LasHeR --yaml_name sutrack_b224

# RGB-Event（VisEvent）
python ./RGBE_workspace/test_rgbe_mgpus.py --script_name sutrack --dataset_name VisEvent --yaml_name sutrack_b224

# RGB-Depth（DepthTrack）
cd Depthtrack_workspace
vot evaluate sutrack_b224
vot analysis sutrack_b224 --nocache
```

### 速度分析
```bash
python tracking/analysis_speed.py
```

## 架构

### 测试流水线
1. `tracking/test.py` - 入口，解析参数
2. `lib/test/evaluation/tracker.py:Tracker.run_sequence()` - 主跟踪循环
3. `lib/test/evaluation/keyframe_loader.py` - 加载关键帧索引（如启用）
4. `lib/test/tracker/{tracker_name}.py` - 跟踪器实现
5. `lib/test/tracker/vlm_engine.py` - VLM 推理（本地或 API）
6. 结果保存至 `./results/{tracker_name}/{dataset}/`

### VLM 跟踪范式

**三种跟踪器**（2026-03 重构）：

1. **qwen_vlm_cognitive（认知跟踪，核心）**
   - 初始化：从初始帧生成语义记忆（appearance/motion/context）
   - 跟踪：上一帧(蓝框) + 当前帧 + 语义记忆 → bbox + 状态判断 + 认知推理
   - Prompt：`cognitive` + `init_memory`
   - 特点：
     - **强制全图搜索**：Prompt 明确要求忽略蓝框位置约束
     - **结构化状态输出**：
       - target_status（6种）：normal, partially_occluded, fully_occluded, out_of_view, disappeared, reappeared
       - environment_status（9种）：normal, low_light, high_light, motion_blur, scene_change, viewpoint_change, scale_change, crowded, background_clutter
     - **认知推理文本**：tracking_evidence（2-4句英文描述）
     - **完整跟踪历史**：保存每帧的状态、环境、推理文本
     - **智能更新策略**：只在目标可见时更新 prev_bbox

2. **qwen_vlm_visual（纯视觉跟踪）**
   - 2帧模式：模板帧(绿框) + 当前帧 → bbox
   - 3帧模式：初始帧(绿框) + 上一帧(蓝框) + 当前帧 → bbox
   - Prompt：`two_image` / `three_image`
   - 特点：纯视觉匹配，无语义记忆

3. **qwen_vlm_hybrid（混合跟踪）**
   - 结合视觉和认知两种模式
   - 支持多种触发策略：keyframe / confidence / hybrid
   - 特点：灵活切换视觉和认知推理

**认知跟踪输出格式**（新）：
```json
{
  "target_status": "normal",
  "environment_status": ["normal"],
  "bbox": [x1, y1, x2, y2],
  "tracking_evidence": "The target is a red sedan with white side stripes...",
  "confidence": 0.9
}
```

**关键特性**：
- VLM 输出结构化状态判断（选择题形式，可训练）
- 全图推理，不受局部搜索窗口限制
- 认知推理文本（支持 SFT/RLHF 训练）
- 可选择性输出 bbox（invisible 状态输出 [0,0,0,0]）

### 核心模块
- `lib/test/tracker/vlm_engine.py` - VLM 推理抽象（本地/API 模式）
- `lib/test/tracker/prompts.py` - 提示词模板管理
- `lib/test/tracker/vlm_utils.py` - 边界框解析、可视化工具
- `lib/test/tracker/qwen_vlm_cognitive.py` - 认知跟踪器（核心）
- `lib/test/tracker/qwen_vlm_visual.py` - 纯视觉跟踪器
- `lib/test/tracker/qwen_vlm_hybrid.py` - 混合跟踪器
- `lib/test/parameter/vlm_common.py` - VLM 公共参数模块（统一配置管理）
- `lib/test/evaluation/keyframe_loader.py` - 关键帧加载
- `lib/models/sutrack/` - SUTrack 模型架构（编码器、解码器、iTPN 骨干网络）

### VLM 推理模式
- **本地模式**：使用 transformers 本地加载 Qwen3-VL/Qwen2.5-VL 模型
- **API 模式**：使用 DashScope API 调用 Qwen3-VL-235B
- 配置位于 `lib/test/parameter/vlm_common.py:MODEL_CONFIGS`
- 支持后缀约定：`default_2f`（2帧）、`local_4b_3f`（本地4B+3帧）、`default_visual_kf`（视觉+关键帧触发）

### 基于关键帧的稀疏跟踪
- 关键帧索引存储在 JSON：`{keyframe_root}/{dataset}/{split}/{seq_name}.json`
- 默认关键帧根目录：`/data/DATASETS_PUBLIC/SOIBench/KeyFrame/scene_changes_resnet/top_10`
- 非关键帧预测填充 NaN 以保持序列长度
- 数据集映射由 `keyframe_loader.py` 中的 `_DATASET_MAP` 处理

## 项目结构

```
lib/
├── test/                    # 测试与评估框架
│   ├── tracker/             # 跟踪器实现（1593 行代码）
│   │   ├── sutrack.py       # SUTrack 基线
│   │   ├── qwen_vlm*.py     # 三种 VLM 跟踪器
│   │   ├── vlm_engine.py    # VLM 推理引擎
│   │   ├── prompts.py       # 提示词模板
│   │   └── vlm_utils.py     # VLM 工具函数
│   ├── parameter/           # 跟踪器配置
│   ├── evaluation/          # 评估框架
│   │   ├── tracker.py       # 主跟踪器包装器
│   │   ├── keyframe_loader.py # 关键帧加载
│   │   └── local.py         # 路径配置
│   └── analysis/            # 分析工具
├── train/                   # 训练框架
├── models/sutrack/          # SUTrack 模型架构
└── config/sutrack/          # 训练配置

tracking/                    # 入口脚本
├── test.py                  # 主测试入口
├── train.py                 # 训练入口
├── analysis_results.py      # 结果分析
└── analysis_speed.py        # 速度分析

SOIBench/vlms/              # VLM 定位框架
├── model_adapters/          # VLM 适配器模式
│   ├── qwen3vl_adapter.py
│   ├── glm46v_adapter.py
│   └── deepseekvl_adapter.py
└── run_grounding.py         # 统一定位入口

experiments/                 # YAML 配置文件
├── sutrack/                 # SUTrack 配置
└── qwen3vl/                 # Qwen3VL 配置
```

## 配置文件

### 路径配置
编辑 `lib/test/evaluation/local.py` 设置：
- 数据集路径（LaSOT、GOT-10K、TrackingNet 等）
- 结果目录
- 关键帧根目录

### 跟踪器参数
VLM 跟踪器配置位于 `lib/test/parameter/qwen_vlm.py`：
- `MODEL_CONFIGS` 字典定义本地/API 模式
- 模型名称、API 端点、temperature、max_tokens

### 关键帧配置
- 简化的加载器，显式的 `_DATASET_MAP`
- 从 `keyframe_config` 改为 `keyframe_root` 参数
- 固定路径格式：`{keyframe_root}/{dataset}/{split}/{seq_name}.json`

## 数据组织

预期目录结构：
```
./data/                      # 数据集
├── lasot/
├── got10k/
├── trackingnet/
├── lasher/                  # RGB-T
├── visevent/                # RGB-E
└── depthtrack/              # RGB-D

./pretrained/itpn/           # 预训练骨干网络
├── fast_itpn_base_clipl_e1600.pt
└── fast_itpn_large_1600e_1k.pt

./checkpoints/train/sutrack/ # 训练好的模型
├── sutrack_b224/
└── sutrack_l384/

/data/DATASETS_PUBLIC/SOIBench/KeyFrame/ # 关键帧索引
├── scene_changes_resnet/
│   ├── top_10/
│   └── top_30/
└── scene_changes_clip/
```

## 开发笔记

### 最近更新（2026-04-02）
- **认知跟踪重大升级**：
  - 新增 `cognitive` prompt，强制全图搜索（忽略蓝框位置约束）
  - 结构化状态输出：6种目标状态 + 9种环境状态（选择题形式）
  - 认知推理文本：tracking_evidence（2-4句中文描述）
  - 完整跟踪历史：保存每帧的状态、环境、推理文本
  - 智能更新策略：只在目标可见时更新 prev_bbox
- 新增 `parse_cognitive_output` 解析器（vlm_utils.py）
- 更新 `InitialMemoryPrompt` 要求中文输出

### 历史更新（2026-02-24）
- 简化关键帧加载：387 → 100 行
- 添加显式的 `_DATASET_MAP` 用于数据集映射
- 从 `keyframe_config` 改为 `keyframe_root` 参数
- 实现三种 VLM 基线跟踪器

### VLM 引擎设计
- 支持本地推理（transformers）和 API 调用（DashScope）
- 统一接口：`VLMEngine.infer(image, prompt)`
- 提示词模板由 `PromptManager` 管理
- 边界框解析处理多种输出格式

### 调试
- 使用 `--debug 1` 标志启用可视化
- 认知跟踪器 debug 输出示例：
  ```
  [Cognitive] Frame 42
    Raw output: {"target_status": "normal", ...}
    Status: normal
    Environment: ['normal']
    Confidence: 0.92
    Evidence: 目标是一辆红色轿车，车身有白色条纹。当前正在向右移动...
  ```
- VSCode 调试配置见 `docs/vscode_debug_guide.md`
- 速度分析：`python tracking/analysis_speed.py`
- 跟踪历史分析：
  ```python
  # 访问完整跟踪历史
  tracker.tracking_history  # List[Dict] 包含每帧的状态、环境、推理文本
  ```

## 重要约定

- 运行任何脚本前必须设置 `PYTHONPATH`
- GOT-10K 需要特殊的结果格式转换
- VLM 跟踪器默认使用基于关键帧的稀疏跟踪
- 非关键帧预测为 NaN（保持序列长度）
- 关键帧加载器中的数据集名称使用特定映射（如 "lasot" → "LaSOT"）
- VLM 推理可能较慢；VLM 跟踪器使用 `--threads 1` 避免内存问题

## 项目规划

本项目分为两个阶段，遵循"先基准后算法"的研究路线：先明确认知跟踪的评测标准和能力差距，再针对性设计算法。

### 阶段一：认知跟踪评测基准（优先）

**目标**：构建专门评测认知跟踪能力的基准数据集和指标体系

**核心任务**：
1. **数据集构建**
   - 参考 SOTVerse/VLTVerse 思路，从现有数据集（LaSOT 等）提取挑战密集子序列
   - 重点关注：遮挡、消失、重现、场景切换、全图搜索等认知场景
   - 逐帧标注目标状态（可见/部分遮挡/完全遮挡/出视野/重现）
   - 构建挑战子序列：>50% 挑战帧，最短 100 帧，重叠率 <0.5

2. **评测指标设计**
   - **状态判断准确率**：模型是否正确识别目标状态（可见/遮挡/消失）
   - **重定位能力**：目标重现后的恢复跟踪速度和准确率
   - **误报率**：目标不可见时错误输出框的比例
   - **全图搜索能力**：不依赖局部窗口的定位准确率
   - **认知推理得分**：综合评估文本输出的语义正确性

3. **基线测试**
   - 在基准上测试现有跟踪算法（传统、VLT、VLM）
   - 量化现有算法的认知能力差距
   - 分析 VLM 在跟踪任务上的潜力和瓶颈

**预期产出**：
- 认知跟踪评测数据集
- 评测工具包和在线平台
- 现有算法的认知能力分析报告

**关键文件位置**：
- LaSOT 标注：`lib/test/evaluation/lasotdataset.py`（`full_occlusion.txt`, `out_of_view.txt`）
- MGIT 数据集：`lib/test/evaluation/videocubedataset.py`
- 评测代码：`lib/test/analysis/extract_results.py`
- SOTVerse 论文：`2204.07414v2.pdf`
- 认知跟踪实现总结：`认知跟踪实现总结.md`
- 认知跟踪输出格式设计：`认知跟踪输出格式设计.md`
- 算法审查与改进建议：`算法审查与改进建议.md`

**MGIT 数据集统计**（可行性验证首选）：
- 路径：`/data/DATASETS_PUBLIC/MGIT`
- 120 个序列，总计 161 万帧，平均每序列 13454 帧（约 7-8 分钟 @30fps）
- absent（目标消失）：23.5%，约 38 万帧
- occlusion（遮挡）：33.3%，约 54 万帧
- shotcut（切镜）：1.6%，约 2.6 万帧
- 极端序列：237（99.5% 遮挡）、242（68.3% absent）、159（96.9% 遮挡）
- 属性标注目录：`/data/DATASETS_PUBLIC/MGIT/attribute/`（absent, occlusion, shotcut, motion, scale, ratio 等）
- 文本标注：`/data/DATASETS_PUBLIC/MGIT/attribute/description/{seq}.json`（逐 action 标注：目标类别、外观、动作、场景、文本描述）
- 注意：`occlusion/032.txt` 第 4717 行有脏数据 'd'，需容错处理
- 数据集分割：test 30 序列，val 15 序列（`/data/DATASETS_PUBLIC/MGIT/data/`）
- 序列列表配置：`lib/test/evaluation/videocube.json`

---

### 阶段二：认知跟踪算法

**目标**：基于阶段一的基准，设计具有认知推理能力的跟踪算法

**核心能力**：
1. **全图搜索**：不受局部窗口限制，VLM 在整张图中推理目标位置
2. **状态推理**：理解并输出目标状态（可见/遮挡/消失/重现）
3. **文本输入输出**：
   - 输入：当前帧 + 文本描述（目标特征、历史状态）
   - 输出：文本状态描述 + bbox（如可见）
4. **自启发推理**：维持对目标的语义理解，做出认知决策

**技术路线**：
- 基于 SUTrack 统一框架
- 利用 Qwen3-VL 等 VLM 的推理能力
- 设计三种范式：双图、三图、记忆库
- 在阶段一的基准上验证效果

**预期产出**：
- 认知跟踪算法实现
- 在基准上的性能提升
- 算法论文

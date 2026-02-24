# VLMTrack Tracker重构指南

## 概述

本次重构将混乱的多个tracker版本精简为**3个核心版本**，并通过模块化设计大幅简化代码。

## 重构目标

### ❌ 重构前的问题

```
lib/test/tracker/
├── qwen3vl.py                    # 基础版本
├── qwen3vl_hybrid.py             # 混合版本1
├── qwen3vl_hybrid_v2.py          # 混合版本2
├── qwen3vl_memory.py             # 记忆库v1
├── qwen3vl_memory_v2.py          # 记忆库v2
├── qwen3vl_memory_v2_optimized.py # 记忆库v2优化版
├── qwen3vl_three_image.py        # 三图版本
└── utils.py                      # 工具函数
```

**问题**:
- ❌ 版本繁多（7个tracker文件）
- ❌ 代码冗余严重（每个文件都重复实现推理、bbox解析等）
- ❌ Prompt硬编码在代码中
- ❌ 关键帧逻辑混在tracker内部
- ❌ 难以维护和扩展

### ✅ 重构后的结构

```
lib/test/tracker/
├── 核心模块（共享）
│   ├── prompts.py          # Prompt配置管理
│   ├── vlm_utils.py        # 通用工具函数
│   └── vlm_engine.py       # VLM推理引擎
│
├── 三个精简tracker
│   ├── qwen_vlm.py         # 版本1: 两图跟踪（基础）
│   ├── qwen_vlm_three.py   # 版本2: 三图跟踪
│   └── qwen_vlm_memory.py  # 版本3: 三图+记忆库
│
└── 旧版本（可选保留）
    └── legacy/             # 移动旧文件到此目录
```

## 三个核心版本

### 版本1: 两图跟踪 (`qwen_vlm.py`)

**最简单的VLM跟踪**

```
跟踪流程:
┌──────────────┐     ┌──────────────┐
│ 模板帧+绿框  │ +   │   当前帧      │  →  VLM  →  BBox预测
└──────────────┘     └──────────────┘
```

**适用场景**:
- 快速验证VLM跟踪能力
- 目标外观变化较小
- 追求简单高效

**代码行数**: ~120行

### 版本2: 三图跟踪 (`qwen_vlm_three.py`)

**固定锚点 + 运动线索**

```
跟踪流程:
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 初始帧+绿框  │ + │ 上一帧+蓝框  │ + │   当前帧      │  →  VLM  →  BBox预测
│  (固定锚点)  │   │  (运动线索)  │   └──────────────┘
└──────────────┘   └──────────────┘
```

**优势**:
- 初始帧提供稳定的目标外观
- 上一帧提供短期运动连续性
- 减少逐帧漂移问题

**代码行数**: ~140行

### 版本3: 三图+记忆库 (`qwen_vlm_memory.py`)

**语义记忆辅助跟踪**

```
初始化:
初始帧+框  →  VLM  →  生成语义记忆 {appearance, motion, context}

跟踪流程:
┌──────────────┐   ┌──────────────┐
│   记忆库      │ + │ 上一帧+蓝框  │ + │   当前帧     │  →  VLM
│  (语义描述)  │   │  (运动线索)  │   └──────────────┘
└──────────────┘   └──────────────┘
                                    ↓
                         ┌─────────────────────┐
                         │ BBox预测 + 更新记忆  │
                         └─────────────────────┘
```

**优势**:
- 语义记忆提供高层次理解
- 一次VLM调用同时输出bbox和记忆更新
- 适应目标外观变化

**代码行数**: ~160行

## 模块化设计

### 1. Prompt配置模块 (`prompts.py`)

**统一管理所有prompt**

```python
from lib.test.tracker.prompts import get_prompt

# 使用示例
prompt = get_prompt("two_image", target_description="a red car")
prompt = get_prompt("three_image", target_description="a person")
prompt = get_prompt("memory_bank", 
                    memory_appearance="red car",
                    memory_motion="moving right")
```

**好处**:
- ✅ Prompt与代码分离
- ✅ 易于调整和实验
- ✅ 支持自定义prompt模板

### 2. 通用工具模块 (`vlm_utils.py`)

**统一的bbox解析、图像处理等**

```python
from lib.test.tracker.vlm_utils import (
    parse_bbox_from_text,  # 从VLM输出解析bbox
    xyxy_to_xywh,          # 格式转换
    draw_bbox,             # 绘制bbox
    numpy_to_base64,       # 图像转base64
)
```

### 3. VLM推理引擎 (`vlm_engine.py`)

**统一的推理接口（本地/API）**

```python
from lib.test.tracker.vlm_engine import VLMEngine

vlm = VLMEngine(params)
output = vlm.infer(images, prompt)  # 自动选择local或API
```

## 代码对比

### 旧版本（冗余）

```python
# qwen3vl.py - 655行
class QWEN3VL:
    def __init__(...):
        # 模型加载
        # 配置API
        # 关键帧逻辑
        # ...
    
    def _build_tracking_prompt(...):
        # 硬编码prompt
        return "..."
    
    def _run_inference(...):
        # 推理逻辑
        ...
    
    # ... 更多重复代码
```

### 新版本（精简）

```python
# qwen_vlm.py - 120行
class QwenVLMTracker:
    def __init__(self, params, dataset_name):
        self.vlm = VLMEngine(params)  # 复用引擎
    
    def track(self, image, info):
        # 构建prompt（从配置读取）
        prompt = get_prompt(self.prompt_name, ...)
        
        # 推理（复用引擎）
        output = self.vlm.infer([template, image], prompt)
        
        # 解析（复用工具）
        bbox = parse_bbox_from_text(output, W, H)
```

## 迁移指南

### Step 1: 创建备份

```bash
# 备份旧文件
mkdir Y:\VLMTrack\lib\test\tracker\legacy
move Y:\VLMTrack\lib\test\tracker\qwen3vl*.py Y:\VLMTrack\lib\test\tracker\legacy\
```

### Step 2: 使用新tracker

修改测试脚本或参数配置:

```python
# 旧版本
--tracker_name qwen3vl_memory_v2 

# 新版本
--tracker_name qwen_vlm_memory
```

### Step 3: 配置prompt（可选）

```python
# lib/test/parameter/your_config.py
def parameters(yaml_name):
    params = ParameterList()
    
    # 选择prompt模板
    params.prompt_name = 'three_image'  # 可选: two_image, three_image, memory_bank
    
    # VLM配置
    params.mode = 'api'
    params.api_model = 'qwen3-vl-235b-a22b-instruct'
    
    # 关键帧配置
    params.use_keyframe = True
    params.keyframe_root = 'Y:/VLMTrack/keyframe_indices'
    
    return params
```

## 优势总结

| 方面 | 旧版本 | 新版本 |
|------|--------|--------|
| **文件数量** | 7个tracker文件 | 3个tracker + 3个工具模块 |
| **代码行数** | ~2000行(重复) | ~600行(精简) |
| **Prompt管理** | 硬编码在代码中 | 统一配置文件 |
| **推理逻辑** | 每个tracker重复 | 共享VLMEngine |
| **工具函数** | 每个tracker重复 | 共享vlm_utils |
| **可维护性** | 低（分散） | 高（集中） |
| **扩展性** | 难（需修改多处） | 易（只改配置） |

## 文件对照表

| 旧文件 | 新文件 | 说明 |
|--------|--------|------|
| `qwen3vl.py` | `qwen_vlm.py` | 基础两图跟踪 |
| `qwen3vl_three_image.py` | `qwen_vlm_three.py` | 三图跟踪 |
| `qwen3vl_memory_v2.py` | `qwen_vlm_memory.py` | 记忆库跟踪 |
| `qwen3vl_hybrid.py` | ❌ 删除 | 功能已整合 |
| `qwen3vl_hybrid_v2.py` | ❌ 删除 | 功能已整合 |
| `qwen3vl_memory.py` | ❌ 删除 | 已被v2替代 |
| `utils.py` | `vlm_utils.py` | 重构并扩展 |

## 后续工作

1. **测试新tracker**: 在多个数据集上验证功能
2. **移除旧文件**: 确认无问题后删除legacy目录
3. **更新文档**: 同步README和使用指南
4. **性能优化**: 根据实际使用情况调优

## 总结

通过本次重构:
- ✅ **代码量减少60%+**
- ✅ **结构更清晰** - 3个核心版本，各有侧重
- ✅ **易于维护** - 模块化设计，修改一处生效全部
- ✅ **易于扩展** - 添加新prompt只需修改配置
- ✅ **职责分离** - tracker专注跟踪，关键帧由evaluation层管理

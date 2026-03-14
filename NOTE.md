# SUTrack 开发笔记

## 2026-01-23: 添加速度分析脚本

### 任务背景
评测代码部分对于指标分析只有AUC等精度指标，没有对速度进行统计。但推理时已经保存了 `time.txt` 文件，需要补充速度分析功能。

### 实现内容

#### 1. 新增文件

**`lib/test/analysis/speed_analysis.py`** - 速度分析核心模块
- `load_time_file()`: 加载 `{seq_name}_time.txt` 文件
- `extract_speed_results()`: 从所有序列提取时间数据
- `compute_speed_statistics()`: 计算速度统计指标
  - 平均 FPS (序列级)
  - 整体 FPS (帧级)
  - FPS 标准差、最小值、最大值、中位数
  - 平均帧处理时间 (ms)
- `print_speed_results()`: 打印详细速度报告
- `print_speed_comparison()`: 打印简化对比表
- `get_per_sequence_fps()`: 获取每序列FPS用于自定义分析

**`tracking/analysis_speed.py`** - 速度分析入口脚本
- 类似 `analysis_results.py` 的使用方式
- 配置 trackers 和 dataset 后直接运行

#### 2. 时间文件格式 (参考 running.py)

保存路径: `{results_dir}/{dataset}/{seq_name}_time.txt`

每行一个浮点数，表示对应帧的处理时间（秒）。

#### 3. 使用方法

```python
# 方法1: 使用入口脚本
# 编辑 tracking/analysis_speed.py 配置 trackers 和 dataset_name
python tracking/analysis_speed.py

# 方法2: 在代码中调用
from lib.test.analysis.speed_analysis import print_speed_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = trackerlist(name='sutrack', parameter_name='sutrack_b224', 
                       dataset_name='lasot', display_name='SUTrack-B224')
dataset = get_dataset('lasot')
print_speed_results(trackers, dataset, report_name='lasot_speed')
```

#### 4. 输出示例

```
===========================================================================
Speed Analysis
===========================================================================
Tracker           | Avg FPS      | Std FPS      | Overall FPS  | Avg Time(ms) |
---------------------------------------------------------------------------
SUTrack-B224      | 45.23        | 5.67         | 44.89        | 22.28        |
===========================================================================
```

### 关键设计决策
1. **排除首帧**: 默认排除第一帧时间（初始化帧通常较慢）
2. **跳过缺失**: 默认跳过没有time.txt的序列
3. **两级FPS统计**: 
   - `Avg FPS`: 每序列FPS的平均值（序列级均值）
   - `Overall FPS`: 总帧数/总时间（帧级均值）

---

## 2026-02-24: VLM长时跟踪项目现状梳理

### 任务背景
用VLM实现长时跟踪，以QwenVL为底座进行稀疏跟踪，目前已有三个基线版本。

### 三个基线跟踪器（lib/test/tracker/）

| 文件 | 类名 | 范式 | 核心特点 |
|---|---|---|---|
| `qwen_vlm.py` | `QwenVLMTracker` | **两图跟踪** | 模板帧(绿框) + 当前帧；动态更新模板 |
| `qwen_vlm_three.py` | `QwenVLMThreeImage` | **三图跟踪** | 初始帧(绿框固定) + 上一帧(蓝框) + 当前帧；减少漂移 |
| `qwen_vlm_memory.py` | `QwenVLMMemory` | **记忆库跟踪** | 初始化生成语义记忆 + 上一帧运动线索；一次VLM调用同时输出bbox+记忆更新 |

### 整体架构
- **VLMEngine** (`vlm_engine.py`): 支持 local(Qwen3VL/Qwen2.5VL) 和 api(DashScope) 两种推理模式
- **PromptManager** (`prompts.py`): 统一管理 `two_image`、`three_image`、`memory_bank`、`init_memory` 四种prompt模板
- **输出格式**: `{"bbox": [x1,y1,x2,y2], "evidence": "...", "confidence": 0.x}`，memory版额外输出 `"state"` 字段
- **稀疏跟踪**: 关键帧控制在evaluation层，通过 `info['is_keyframe']` 传入各tracker方法

---

## 2026-02-24: 简化关键帧加载逻辑

### 背景
所有关键帧索引已预先计算完毕，路径格式固定为：
`/data/DATASETS_PUBLIC/SOIBench/KeyFrame/{model}/{threshold}/{dataset}/{split}/{seq_name}.jsonl`

旧的 `KeyframeIndexLoader` 类（387行）存在过度设计：dict/string 双格式配置、旧版路径兼容、全局单例等。

### 改动文件

| 文件 | 改动 |
|---|---|
| `lib/test/evaluation/keyframe_loader.py` | 删除类，改为两个函数：`load_keyframe_indices()` + `_parse_index_file()`；新增 `_DATASET_MAP` 显式映射表；387行→100行 |
| `lib/test/evaluation/local.py` | `keyframe_root` 从 dict → 完整字符串路径（含model/threshold） |
| `lib/test/parameter/qwen_vlm*.py`（3个） | `env_settings()` 统一为 `env`；keyframe配置块10行→1行 |
| `lib/test/evaluation/tracker.py` | 参数名 `keyframe_config` → `keyframe_root` |
| `docs/keyframe_config_guide.md` | 同步更新文档 |

### 关键设计
- **`keyframe_root` = 完整前缀路径**，切换模型/阈值只改 `local.py` 一行字符串
- **`_DATASET_MAP` 显式映射**，替代字符串解析，新增数据集只需加一行
- **无全局单例/缓存**，推理时每序列顺序加载一次，无共享状态

---

## 2026-03-08: 全局代码审查 - VLM跟踪部分

### 项目整体架构

```
VLMTrack/
├── lib/
│   ├── models/sutrack/          # SUTrack深度学习跟踪器(骨干网络)
│   ├── test/
│   │   ├── tracker/             # 所有Tracker实现
│   │   │   ├── basetracker.py   # 基类
│   │   │   ├── sutrack.py       # 传统深度学习跟踪器
│   │   │   ├── qwen_vlm.py      # VLM基础两图跟踪器
│   │   │   ├── qwen_vlm_three.py  # VLM三图跟踪器
│   │   │   ├── qwen_vlm_memory.py # VLM记忆库跟踪器
│   │   │   ├── vlm_engine.py    # VLM推理引擎(local/api双模式)
│   │   │   ├── vlm_utils.py     # 工具函数(解析/绘图/API)
│   │   │   └── prompts.py       # Prompt模板管理
│   │   ├── evaluation/
│   │   │   ├── tracker.py       # Tracker评测封装(含稀疏跟踪逻辑)
│   │   │   ├── running.py       # 运行控制(单线程/多进程)
│   │   │   ├── keyframe_loader.py # 关键帧索引加载
│   │   │   └── local.py         # 本地路径配置
│   │   └── parameter/           # 各tracker的参数文件
└── tracking/test.py             # 评测入口脚本
```

### VLM跟踪部分核心模块审查

#### 1. VLMEngine (`vlm_engine.py`) - 推理引擎
- ✅ 设计清晰，local/api双模式统一接口
- ✅ local模式自动区分Qwen3VL和Qwen2.5VL
- ✅ flash_attention_2加载失败有fallback
- ⚠️ **问题**: API模式每次调用都重新创建`OpenAI`客户端对象，性能浪费
- ⚠️ **问题**: `max_new_tokens=256` 硬编码，对于复杂记忆输出可能不够
- ⚠️ **问题**: local模式长时间运行可能有显存碎片，可考虑定期`empty_cache()`

#### 2. 三个VLM Tracker (`qwen_vlm*.py`) - 跟踪器
- ✅ 架构统一，均继承BaseTracker，接口规范
- ✅ 两图/三图/记忆库三种范式梯度覆盖不同复杂度需求
- ⚠️ **共性问题**: 解析失败时统一返回`[0,0,0,0]`，评测时会计算为error结果而非skip
- ⚠️ **qwen_vlm.py**: 动态更新模板（每帧覆盖）存在漂移累积风险
- ⚠️ **qwen_vlm_memory.py**: 初始化VLM parse失败的兜底逻辑直接用`language_description`，而该字段来自`init_nlp`，有些数据集可能为空
- ⚠️ **三个文件都有**: MODEL_CONFIGS 字典完全重复，应抽为公共模块

#### 3. PromptManager (`prompts.py`) - Prompt管理
- ✅ 面向对象设计，可扩展性好
- ✅ 支持运行时动态注册新Prompt
- ⚠️ **潜在问题**: `PromptManager._prompts` 是类变量共享，多进程评测时若存在并发注册会有竞争条件（当前无并发注册，暂时安全）
- ✅ `two_image/three_image/memory_bank/init_memory` 四种prompt覆盖了所有使用场景

#### 4. 稀疏跟踪机制 (`tracker.py` + `keyframe_loader.py`)
- ✅ 关键帧控制在evaluation层，tracker无需感知（低耦合）
- ✅ 非关键帧填充NaN bbox，保持输出长度与总帧数一致
- ⚠️ **问题**: NaN bbox写入txt文件后，评测脚本需要能正确处理NaN（需验证各dataset评测代码对NaN的兼容性）
- ⚠️ **问题**: `keyframe_loader.py` 第129行每次都print，大规模评测时输出过多
- ✅ `_DATASET_MAP` 显式映射设计干净，新增数据集成本极低

#### 5. vlm_utils.py - 工具函数
- ✅ `parse_bbox_from_text()` 鲁棒：JSON→正则两级fallback
- ✅ `convert_to_pixel_bbox()` 自动识别0-1/0-1000/像素三种格式
- ⚠️ **潜在问题**: 当bbox四个坐标值都恰好落在(0,1)范围但本意是像素坐标时会误判为归一化坐标（概率极低，但理论存在）
- ✅ `call_vlm_api()` 有重试机制

### 后续重点改进方向

1. **MODEL_CONFIGS去重**: 三个parameter文件中的MODEL_CONFIGS完全相同，建议抽取到 `lib/test/parameter/vlm_common.py`
2. **API客户端复用**: `VLMEngine._setup_api()` 阶段创建并缓存`OpenAI`客户端
3. **NaN bbox评测兼容性**: 确认各数据集评测代码能正确跳过NaN行
4. **Parse失败处理**: 考虑返回上一帧bbox而非`[0,0,0,0]`（last-frame fallback）
5. **关键帧加载日志**: 将`print`改为`logging.debug`，减少大规模评测的输出噪音

---

## 2026-03-08: VLM跟踪代码重构

### 改动概览

将原有三个VLM Tracker（`qwen_vlm`/`qwen_vlm_three`/`qwen_vlm_memory`）重构为体系更清晰的三类Tracker：

| 新Tracker类 | 中文定义 | 代替原文件 |
|------------|---------|---------|
| `QwenVLMVisual` | 纯视觉跟踪 | qwen_vlm + qwen_vlm_three (合并) |
| `QwenVLMCognitive` | 认知跟踪 | qwen_vlm_memory (重构) |
| `QwenVLMHybrid` | 混合跟踪 | 全新 (SUTrack + VLM) |

### 新文件

**Tracker层** (`lib/test/tracker/`)

- `qwen_vlm_visual.py` — 纯视觉，`num_frames=2/3` 统一控制两图/三图
- `qwen_vlm_cognitive.py` — 认知，语义记忆+运动线索
- `qwen_vlm_hybrid.py` — 混合，SUTrack基础 + VLM按需校正

**Parameter层** (`lib/test/parameter/`)

- `vlm_common.py` — 共用 MODEL_CONFIGS + yaml_name 后缀解析器 `parse_yaml_name()`
- `qwen_vlm_visual.py`、`qwen_vlm_cognitive.py`、`qwen_vlm_hybrid.py`

### 削除文件

`qwen_vlm.py`、`qwen_vlm_three.py`、`qwen_vlm_memory.py`（tracker + parameter 各3个，共6个）

### 关键设计

#### 纯视觉 / 认知跟踪
- **Parse失败改为 last-frame fallback**（保持 `self.state` 不变），不再返回零框
- `language_description` 为 `None` 时兜底为 `"the target object"`

#### 混合跟踪器 (`QwenVLMHybrid`)
- `vlm_mode`: `'visual'` 或 `'cognitive'`，选择 VLM 推理范式
- `trigger_mode`: 三种触发策略：
  - `'keyframe'` — 外部关键帧索引控制
  - `'confidence'` — SUTrack 置信度 < `conf_threshold` 时自动触发
  - `'hybrid'` — 两者取其一
- VLM 校正成功后**立即重置 SUTrack 模板**，防止漂移累积
- 无 NaN 填充，全程连续输出 bbox

#### yaml_name 后缀约定
```
qwen_vlm_visual:    {model}[_{N}f]
                    e.g. default_2f / local_4b_3f
qwen_vlm_hybrid:    {model}_{vlm_mode}_{trigger}
                    e.g. default_visual_kf / local_4b_cognitive_conf
```

#### 基础设施改进
- `vlm_engine.py`: API 客户端缓存（`self.client`），避免每帧重建
- `vlm_utils.py`: `call_vlm_api()` 接受预建 `client` 参数
- `keyframe_loader.py`: 移除 per-sequence print，只保留 warning 级别输出

### 使用示例

```bash
# 纯视觉 - API两图
python tracking/test.py qwen_vlm_visual default --dataset_name lasot

# 纯视觉 - 本地4B三图
python tracking/test.py qwen_vlm_visual local_4b_3f --dataset_name lasot

# 认知跟踪
python tracking/test.py qwen_vlm_cognitive default --dataset_name lasot

# 混合 - 关键帧触发 + 视觉VLM
python tracking/test.py qwen_vlm_hybrid default_visual_kf --dataset_name lasot

# 混合 - 置信度触发 + 认知VLM
python tracking/test.py qwen_vlm_hybrid default_cognitive_conf --dataset_name lasot
```

---

## 2026-03-13: Prompt工程优化 + 代码质量改进

### 改动概览

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `lib/test/tracker/prompts.py` | 重写 | 四个prompt模板全面升级 |
| `lib/test/tracker/vlm_engine.py` | 参数化 | `max_new_tokens` 硬编码→可配置 |
| `lib/test/tracker/qwen_vlm_cognitive.py` | 重构 | `_generate_initial_memory` 使用 `parse_memory_state` 统一解析，移除重复import |
| `lib/test/tracker/qwen_vlm_hybrid.py` | 重构 | 同上，移除 `re`/`json`/`dict_to_str` 重复import |

---

### 1. Prompt工程升级 (`prompts.py`)

#### 改进原则

| 原则 | 说明 |
|---|---|
| **任务分解** | 用 Step 1/2/3 引导VLM逐步推理（轻量CoT） |
| **反例引导** | 明确列出何时输出 `[0,0,0,0]`（遮挡/出框） |
| **鲁棒性** | 覆盖尺度变化、光照变化、部分遮挡等常见退化场景 |
| **格式约束** | 强调"Respond with ONLY this JSON"，减少解析失败 |

#### 四个模板具体改动

**`two_image` (两图跟踪)**
- 新增 Step 1/2/3 推理步骤：先分析目标特征 → 再定位 → 处理边界情况
- 强调"寻找 SAME INSTANCE，不被相似物体迷惑"
- 多候选时按特征相似度优先选择，考虑位置连续性

**`three_image` (三图跟踪)**
- 明确 Image 1 = 外观参考（权威），Image 2 = 仅用于运动方向估计
- 新增运动预测步骤：方向 + 速度 + 在 Image 3 中的预测位置
- 搜索策略：先近后远，外观优先于位置

**`memory_bank` (记忆库跟踪)**
- 强调记忆容忍度：颜色/纹理因光照漂移、尺寸因深度变化
- `state.motion` 要求输出方向和速度估计
- `state.appearance` 需注明相对于原记忆的变化

**`init_memory` (初始记忆生成)**
- 明确引导生成 discriminative features（有利于长时重识别）
- 提示关注：主次颜色、纹理类型、独特标记、尺寸比例
- 场景类型引导：室内/室外、道路/田野等

---

### 2. `max_new_tokens` 参数化 (`vlm_engine.py`)

**问题**: 原来硬编码 `max_new_tokens=256`，对于包含 `state` 字段的 memory_bank 输出可能截断

**改动**: 默认从 256 提升到 512，并可通过 `params.max_new_tokens` 按需配置。

---

### 3. `_generate_initial_memory` 统一重构 (cognitive + hybrid)

**问题**: 两个文件各自用 `re.sub` + `json.loads` 手动解析，与 `parse_memory_state` 逻辑重复

**改动**: 统一调用 `parse_memory_state(output)`，失败时回退到语言描述。移除两个文件中的 `import re, json, dict_to_str`。

---

### 研究思路与后续规划

#### 当前三个跟踪范式对比

| 范式 | 强项 | 弱项 | 适用场景 |
|---|---|---|---|
| Visual (2-frame) | 简单快速 | 长时漂移 | 短序列、实时性要求高 |
| Visual (3-frame) | 利用运动线索 | 无语义理解 | 运动规律性强的序列 |
| Cognitive | 语义记忆鲁棒 | 记忆误更新风险 | 外观变化大、长时遮挡 |
| Hybrid (SUTrack+VLM) | 连续+语义双保障 | 系统复杂度高 | 高精度需求场景 |

#### 下一步研究方向

1. **Prompt消融实验**: 对比新旧prompt在LaSOT/TNL2K上的精度差异，验证CoT引导的效果
2. **记忆更新策略**: Cognitive模式中，当 `confidence < 0.5` 时冻结 memory state，避免错误传播
3. **VLM置信度反馈**: 将VLM的 `confidence` 字段反馈给Hybrid触发器（动态阈值替代固定阈值）
4. **模型规模消融**: 4B vs 8B vs 235B在跟踪精度和速度上的权衡分析

---

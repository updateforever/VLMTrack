# VLMTrack 关键帧索引优化方案

## 概述

本优化将关键帧索引加载逻辑从tracker内部移至evaluation层（`lib/test/evaluation/tracker.py`），使代码结构更清晰，职责分离更明确。

## 核心改进

### 1. **职责分离**
- **Evaluation层** (`tracker.py`): 负责关键帧索引的加载和判断
- **Tracker层** (具体tracker): 只负责VLM推理，不关心是否为关键帧

### 2. **统一的索引加载接口**
- 新增 `lib/test/evaluation/keyframe_loader.py` 模块
- 根据数据集名和序列名自动查找索引文件
- 支持多种文件格式和目录结构

### 3. **灵活的配置方式**
- 通过params配置启用/禁用关键帧模式
- 支持全局keyframe_root配置

## 文件结构

```
VLMTrack/
├── lib/test/evaluation/
│   ├── keyframe_loader.py          # 【新增】关键帧索引加载模块
│   └── tracker.py                   # 【修改】集成关键帧逻辑
├── lib/test/tracker/
│   ├── qwen3vl_memory_v2_optimized.py  # 【新增】优化版tracker示例
│   └── ...
├── lib/test/parameter/
│   └── qwen3vl_memory_v2_example.py    # 【新增】配置示例
├── keyframe_indices/                # 【建议】索引文件存放目录
│   ├── lasot/
│   │   ├── airplane-1.json
│   │   └── ...
│   ├── tnl2k/
│   └── videocube/
└── demo.json                        # 索引文件格式示例
```

## 索引文件格式

支持以下三种格式：

### 格式1: demo.json格式（单序列）
```json
{
    "sequence": "airplane-1",
    "total_frames": 1000,
    "key_frames": [0, 10, 20, 30, ...]
}
```

### 格式2: 多序列索引文件
```json
{
    "airplane-1": {
        "key_frames": [0, 10, 20, ...]
    },
    "bear-1": {
        "key_frames": [0, 15, 30, ...]
    }
}
```

### 格式3: 纯数组格式
```json
[0, 10, 20, 30, 40, 50, ...]
```

## 目录组织方式

索引文件查找优先级（按顺序）：

1. `{keyframe_root}/{dataset_name}/{seq_name}.json`
   - 例: `keyframe_indices/lasot/airplane-1.json`

2. `{keyframe_root}/{dataset_name}/{category}/{seq_name}.json`
   - 例: `keyframe_indices/lasot/airplane/airplane-1.json`

3. `{keyframe_root}/{seq_name}.json`
   - 例: `keyframe_indices/airplane-1.json`

## 使用方法

### 步骤1: 准备索引文件

```bash
# 创建目录结构
mkdir -p Y:/VLMTrack/keyframe_indices/lasot
mkdir -p Y:/VLMTrack/keyframe_indices/tnl2k
mkdir -p Y:/VLMTrack/keyframe_indices/videocube

# 将您的demo.json复制并重命名
# 例如，对于lasot的airplane-1序列
cp Y:/VLMTrack/demo.json Y:/VLMTrack/keyframe_indices/lasot/airplane-1.json
```

### 步骤2: 配置参数文件

在您的parameter文件中（例如 `lib/test/parameter/qwen3vl_memory_v2.py`）:

```python
def parameters(yaml_name: str):
    params = ParameterList()
    
    # ========== 启用关键帧稀疏推理 ==========
    params.use_keyframe = True
    params.keyframe_root = 'Y:/VLMTrack/keyframe_indices'
    
    # ========== VLM配置 ==========
    params.mode = 'api'
    params.api_model = 'qwen3-vl-235b-a22b-instruct'
    # ... 其他配置
    
    return params
```

### 步骤3: 运行测试

```bash
# 运行跟踪测试
python tracking/test.py --tracker_name qwen3vl_memory_v2 \
                        --tracker_param default \
                        --dataset_name lasot \
                        --sequence airplane-1
```

## 代码流程

### Evaluation层 (tracker.py)

```python
# 1. 初始化后加载关键帧索引
if tracker.params.use_keyframe:
    keyframe_indices = load_keyframe_indices(
        dataset_name=self.dataset_name,
        seq_name=seq.name,
        index_root=tracker.params.keyframe_root
    )

# 2. 跟踪循环中判断是否为关键帧
for frame_num in range(1, len(seq.frames)):
    if use_keyframe and frame_num not in keyframe_indices:
        # 非关键帧：跳过VLM推理
        out = None
    else:
        # 关键帧：调用tracker
        out = tracker.track(image, info)
```

### Tracker层 (qwen3vl_memory_v2_optimized.py)

```python
def track(self, image, info: dict = None):
    """
    只负责VLM推理，不关心是否为关键帧
    evaluation层会自动跳过非关键帧
    """
    # 直接进行VLM推理
    prompt = self._tracking_with_state_prompt()
    output = self._run_inference([prev_with_box, image], prompt)
    
    # 解析并返回结果
    bbox_xyxy, new_state = self._parse_tracking_output(output, W, H)
    return {"target_bbox": pred_bbox}
```

## 优势对比

### 优化前（原qwen3vl_memory_v2.py）
```python
class QWEN3VL_Memory_V2:
    def __init__(self, params, dataset_name):
        # ❌ tracker内部需要处理关键帧索引
        self.keyframe_indices = None
        self.keyframe_root = getattr(params, 'keyframe_root', None)
        
    def initialize(self, image, info):
        # ❌ 在initialize中加载索引
        if self.use_keyframe and self.keyframe_root:
            self.keyframe_indices = read_keyframe_indices(...)
    
    def track(self, image, info):
        # ❌ tracker内部判断是否为关键帧
        if self.use_keyframe and self.keyframe_indices:
            if self.frame_id not in self.keyframe_indices:
                return None
        # 进行VLM推理...
```

**问题**:
- tracker职责混乱（既要做推理，又要管理关键帧）
- 每个tracker都需要重复实现关键帧逻辑
- 难以统一管理和修改

### 优化后

```python
# lib/test/evaluation/tracker.py
class Tracker:
    def _track_sequence(self, tracker, seq, init_info):
        # ✅ evaluation层统一加载索引
        keyframe_indices = load_keyframe_indices(...)
        
        for frame_num in ...:
            # ✅ evaluation层统一判断
            if frame_num not in keyframe_indices:
                out = None
            else:
                out = tracker.track(image, info)

# lib/test/tracker/qwen3vl_memory_v2_optimized.py
class QWEN3VL_Memory_V2:
    def track(self, image, info):
        # ✅ tracker只负责VLM推理
        # 不需要关心是否为关键帧
        output = self._run_inference(...)
        return {"target_bbox": pred_bbox}
```

**优势**:
- ✅ 职责清晰：evaluation管理流程，tracker专注推理
- ✅ 代码复用：所有tracker共享同一套关键帧逻辑
- ✅ 易于维护：修改关键帧逻辑只需改一处

## 扩展性

### 添加新的索引格式

编辑 `lib/test/evaluation/keyframe_loader.py`:

```python
def _parse_index_file(self, path: Path, seq_name: str):
    # 添加新格式的解析逻辑
    if 'your_new_format' in data:
        # 解析逻辑
        return your_parsed_indices
```

### 添加新的查找路径

编辑 `keyframe_loader.py`:

```python
def _get_candidate_paths(self, root: Path, dataset_name: str, seq_name: str):
    candidates = []
    # 添加新的路径模式
    candidates.append(root / 'your_new_pattern' / f"{seq_name}.json")
    return candidates
```

## 测试验证

### 1. 测试索引加载

```python
from lib.test.evaluation.keyframe_loader import load_keyframe_indices

# 测试单个序列
indices = load_keyframe_indices(
    dataset_name='lasot',
    seq_name='airplane-1',
    index_root='Y:/VLMTrack/keyframe_indices'
)
print(f"Loaded {len(indices)} keyframes: {sorted(list(indices))[:10]}...")
```

### 2. 测试完整流程

```bash
# 启用debug模式查看日志
python tracking/test.py --tracker_name qwen3vl_memory_v2_optimized \
                        --tracker_param default \
                        --dataset_name lasot \
                        --debug 1
```

预期输出：
```
[KeyframeLoader] Loaded 150 keyframes for airplane-1 from airplane-1.json
[Tracker] Loaded 150 keyframes for airplane-1 (15.0% of total frames)
[MemoryV2] Frame 1[KF]: ...
[MemoryV2] Frame 10[KF]: ...
(非关键帧被自动跳过)
```

## 迁移指南

如果您现有的tracker使用了内部关键帧逻辑，可以按以下步骤迁移：

### Step 1: 移除tracker内部的关键帧代码

```python
# 删除或注释掉
# self.keyframe_indices = None
# self.use_keyframe = getattr(params, 'use_keyframe', False)
# self.keyframe_root = getattr(params, 'keyframe_root', None)

# def initialize():
#     if self.use_keyframe:
#         self.keyframe_indices = read_keyframe_indices(...)
```

### Step 2: 简化track方法

```python
def track(self, image, info: dict = None):
    # 移除关键帧判断
    # if self.use_keyframe and self.frame_id not in self.keyframe_indices:
    #     return None
    
    # 直接进行推理
    output = self._run_inference(...)
    return {"target_bbox": ...}
```

### Step 3: 更新参数配置

```python
# lib/test/parameter/your_tracker.py
params.use_keyframe = True
params.keyframe_root = 'Y:/VLMTrack/keyframe_indices'
```

## 常见问题

### Q1: 如果找不到索引文件会怎样？

**A**: 系统会输出警告并自动降级为密集跟踪（每帧都调用tracker）。

### Q2: 可以在运行时禁用关键帧模式吗？

**A**: 可以，在params中设置 `params.use_keyframe = False` 即可。

### Q3: 如何批量生成索引文件？

**A**: 可以使用场景变化检测工具（如CLIP-based方法）批量生成，格式参考demo.json。

### Q4: 索引文件必须放在特定目录吗？

**A**: 不必须，只要在`keyframe_root`指定的路径下按照查找规则组织即可。

## 总结

通过此次优化：
1. ✅ **代码更清晰**：evaluation管流程，tracker专注推理
2. ✅ **易于扩展**：添加新tracker无需重复实现关键帧逻辑
3. ✅ **灵活配置**：统一的配置接口，支持多种索引格式
4. ✅ **向后兼容**：未启用关键帧时与原逻辑完全一致

建议将所有VLM tracker迁移到此架构，以获得更好的代码可维护性。

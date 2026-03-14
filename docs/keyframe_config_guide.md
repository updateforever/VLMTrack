# 关键帧路径配置指南

## 📁 路径结构（固定格式）

```
/data/DATASETS_PUBLIC/SOIBench/KeyFrame/    ← 固定根目录
├── scene_changes_resnet/                   ← 检测模型
│   ├── top_10/                             ← 关键帧比例
│   │   ├── lasot/
│   │   │   ├── val/
│   │   │   │   └── airplane-1.jsonl
│   │   │   └── test/
│   │   ├── mgit/                           ← videocube 数据集对应目录
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── tnl2k/
│   │       └── test/
│   ├── top_30/
│   └── top_80/
└── scene_changes_clip/
    ├── top_10/
    └── top_30/
```

**完整路径格式**: `{keyframe_root}/{dataset}/{split}/{seq_name}.jsonl`

其中 `keyframe_root` 已包含模型和阈值层级，例如：
```
/data/DATASETS_PUBLIC/SOIBench/KeyFrame/scene_changes_resnet/top_10
```

---

## ⚙️ 配置方式（唯一入口）

编辑 `lib/test/evaluation/local.py`，修改 `keyframe_root` 的最后两级路径即可切换模型/阈值：

```python
# lib/test/evaluation/local.py
settings.keyframe_root = '/data/DATASETS_PUBLIC/SOIBench/KeyFrame/scene_changes_resnet/top_10'
```

切换示例：
```python
# 使用 CLIP 模型
settings.keyframe_root = '/data/DATASETS_PUBLIC/SOIBench/KeyFrame/scene_changes_clip/top_10'

# 使用 top_30 阈值
settings.keyframe_root = '/data/DATASETS_PUBLIC/SOIBench/KeyFrame/scene_changes_resnet/top_30'
```

三个 tracker 的参数文件（`qwen_vlm.py` / `qwen_vlm_memory.py` / `qwen_vlm_three.py`）
均通过以下方式读取，**无需修改参数文件**：

```python
params.keyframe_root = getattr(env, 'keyframe_root', '')
```

---

## 🔍 dataset_name → 目录 映射表

加载时，`keyframe_loader.py` 中的 `_DATASET_MAP` 决定如何将 `dataset_name` 转换为目录路径：

| 注册的 dataset_name         | dataset 目录 | split  |
|-----------------------------|-------------|--------|
| `lasot`                     | lasot       | test   |
| `lasot_test`                | lasot       | test   |
| `lasot_val`                 | lasot       | val    |
| `videocube` / `videocube_test` | mgit     | test   |
| `videocube_val`             | mgit        | val    |
| `videocube_test_tiny`       | mgit        | test   |
| `videocube_val_tiny`        | mgit        | val    |
| `tnl2k` / `tnl2k_test`     | tnl2k       | test   |
| `tnl2k_val`                 | tnl2k       | val    |
| `mgit` / `mgit_test`        | mgit        | test   |
| `mgit_val`                  | mgit        | val    |

> 新增数据集时，只需在 `keyframe_loader.py` 的 `_DATASET_MAP` 中添加一行即可。

---

## 📝 索引文件格式

支持两种 JSON 格式（优先级依次）：

```json
// 格式1（标准）：
{"key_frames": [0, 8, 17, 25, 34, ...]}

// 格式2（纯数组）：
[0, 8, 17, 25, 34, ...]
```

---

## 🔧 loader 使用方式

```python
from lib.test.evaluation.keyframe_loader import load_keyframe_indices

indices = load_keyframe_indices(
    dataset_name='lasot_test',
    seq_name='airplane-1',
    keyframe_root='/data/DATASETS_PUBLIC/SOIBench/KeyFrame/scene_changes_resnet/top_10',
)
# → {0, 8, 17, 25, ...}  or None
```

返回值为 `Set[int]`（帧号集合，0-indexed），找不到文件时返回 `None`。

---

## 🚀 快速开始

1. 在 `local.py` 中设置 `keyframe_root`（一行搞定）
2. 确保 tracker 参数中 `use_keyframe = True`（三个 tracker 默认已开启）
3. 运行：

```bash
python tracking/test.py qwen_vlm api --dataset_name lasot_test --sequence airplane-1
```

日志输出示例：
```
[KeyframeLoader] Loaded 45 keyframes for 'airplane-1'
[Tracker] Loaded 45 keyframes for airplane-1 (9.8% of total frames)
```

# 关键帧路径配置指南

## 📁 路径结构

```
/data/DATASETS_PUBLIC/SOIBench_f/              # 基础路径 (root)
├── scene_changes_resnet/                      # 筛选模型 (model)
│   ├── top_10/                               # 筛选阈值 (threshold)
│   │   ├── lasot/                            # 数据集
│   │   │   ├── train/                       # split
│   │   │   ├── val/
│   │   │   └── test/
│   │   │       └── airplane-1.jsonl         # 序列文件
│   │   ├── tnl2k/
│   │   └── videocube/
│   ├── top_30/
│   └── top_80/
└── scene_changes_clip/                        # 另一个筛选模型
    ├── top_10/
    └── top_30/
```

**完整路径**: `{root}/{model}/{threshold}/{dataset}/{split}/{seq_name}.jsonl`

---

## ⚙️ 配置方式

### 方式1: 在env_settings中配置（推荐）

编辑 `lib/test/evaluation/local.py`:

```python
# =========== Keyframe Configuration ===========
settings.keyframe_root = {
    'root': '/data/DATASETS_PUBLIC/SOIBench_f',
    'model': 'scene_changes_clip',       # 可选: scene_changes_resnet
    'threshold': 'top_10',               # 可选: top_30, top_80
}
```

### 方式2: 在参数文件中直接指定

编辑 `lib/test/parameter/qwen_vlm.py`:

```python
params.keyframe_root = {
    'root': '/data/DATASETS_PUBLIC/SOIBench_f',
    'model': 'scene_changes_resnet',     # 使用不同的模型
    'threshold': 'top_30',               # 使用不同的阈值
}
```

### 旧版配置（向后兼容）

```python
# 仍然支持字符串路径
params.keyframe_root = '/path/to/keyframe_indices'
```

---

## 🔍 Dataset名称解析

框架自动从`dataset_name`解析出`dataset`和`split`:

| 注册的dataset_name | 解析为dataset | 解析为split |
|--------------------|--------------|------------|
| `lasot` | lasot | test (默认) |
| `lasot_test` | lasot | test |
| `lasot_val` | lasot | val |
| `videocube_test` | videocube | test |
| `videocube_val` | videocube | val |
| `tnl2k` | tnl2k | test (默认) |

---

## 📝 使用示例

### 示例1: 使用scene_changes_clip模型

```python
# local.py配置
settings.keyframe_root = {
    'root': '/data/DATASETS_PUBLIC/SOIBench_f',
    'model': 'scene_changes_clip',
    'threshold': 'top_10',
}
```

**运行测试**:
```bash
python tracking/test.py qwen_vlm api --dataset_name lasot_test --sequence airplane-1
```

**查找路径**:
```
/data/DATASETS_PUBLIC/SOIBench_f/
  scene_changes_clip/
    top_10/
      lasot/
        test/
          airplane-1.jsonl  # ✅ 找到
```

### 示例2: 使用scene_changes_resnet模型 + top_30阈值

```python
settings.keyframe_root = {
    'root': '/data/DATASETS_PUBLIC/SOIBench_f',
    'model': 'scene_changes_resnet',
    'threshold': 'top_30',
}
```

**运行测试**:
```bash
python tracking/test.py qwen_vlm_three api --dataset_name videocube_val --sequence seq001
```

**查找路径**:
```
/data/DATASETS_PUBLIC/SOIBench_f/
  scene_changes_resnet/
    top_30/
      videocube/
        val/
          seq001.jsonl  # ✅ 找到
```

---

## 🎯 配置参数说明

### root
- **含义**: 关键帧索引文件的基础路径
- **示例**: `/data/DATASETS_PUBLIC/SOIBench_f`

### model
- **含义**: 场景变化检测模型
- **可选值**:
  - `scene_changes_clip` - 使用CLIP模型检测
  - `scene_changes_resnet` - 使用ResNet模型检测
- **默认值**: `scene_changes_clip`

### threshold  
- **含义**: 关键帧筛选阈值级别
- **可选值**:
  - `top_10` - 保留10%的关键帧（稀疏）
  - `top_30` - 保留30%的关键帧（中等）
  - `top_80` - 保留80%的关键帧（密集）
- **默认值**: `top_10`

---

## 🔧 路径查找逻辑

### 新版结构查找

1. **解析dataset_name**
   - `lasot_test` → dataset=`lasot`, split=`test`
   - `videocube_val` → dataset=`videocube`, split=`val`

2. **构建候选路径**
   ```
   {root}/{model}/{threshold}/{dataset}/{split}/{seq_name}.jsonl
   {root}/{model}/{threshold}/{dataset}/{split}/{seq_name}.json
   ```

3. **如果dataset_name没有split后缀，尝试其他split**
   ```
   {root}/{model}/{threshold}/{dataset}/val/{seq_name}.jsonl
   {root}/{model}/{threshold}/{dataset}/train/{seq_name}.jsonl
   ```

### 旧版结构查找（向后兼容）

```
{root}/{dataset}/{seq_name}.jsonl
{root}/{dataset}/{seq_name}.json
{root}/{dataset}/{category}/{seq_name}.jsonl
```

---

## ✨ 优势

- ✅ **灵活的模型选择**: 支持多种场景变化检测模型
- ✅ **可配置的阈值**: 轻松切换不同的筛选密度
- ✅ **清晰的split管理**: 自动解析train/val/test
- ✅ **向后兼容**: 仍支持旧版简单路径
- ✅ **易于扩展**: 添加新模型或阈值只需修改配置

---

## 🔄 切换不同配置

### 测试不同模型

```python
# CLIP模型
params.keyframe_root = {
    'root': '/data/DATASETS_PUBLIC/SOIBench_f',
    'model': 'scene_changes_clip',
    'threshold': 'top_10',
}

# ResNet模型
params.keyframe_root = {
    'root': '/data/DATASETS_PUBLIC/SOIBench_f',
    'model': 'scene_changes_resnet',
    'threshold': 'top_10',
}
```

### 测试不同阈值

```python
# 稀疏跟踪 (10%)
params.keyframe_root['threshold'] = 'top_10'

# 中等密度 (30%)
params.keyframe_root['threshold'] = 'top_30'

# 密集跟踪 (80%)
params.keyframe_root['threshold'] = 'top_80'
```

---

## 📌 注意事项

1. **文件格式**: 优先查找`.jsonl`文件，其次是`.json`文件
2. **dataset注册**: 确保dataset已在`datasets.py`中正确注册
3. **路径存在性**: 确保配置的路径在文件系统中确实存在
4. **权限**: 确保有读取关键帧索引文件的权限

---

## 🚀 快速开始

1. **配置keyframe_root**:
   ```python
   # lib/test/evaluation/local.py
   settings.keyframe_root = {
       'root': '/data/DATASETS_PUBLIC/SOIBench_f',
       'model': 'scene_changes_clip',
       'threshold': 'top_10',
   }
   ```

2. **运行tracker**:
   ```bash
   python tracking/test.py qwen_vlm api \
       --dataset_name lasot_test \
       --debug 1
   ```

3. **检查日志**:
   ```
   [KeyframeLoader] Loaded 45 keyframes for airplane-1 from airplane-1.jsonl
   ```

**配置完成！** 🎉

# SOIBench 场景变化与关键帧提取项目文档

**SOIBench: Unsupervised Scene Change Detection & Keyframe Extraction**

本项目旨在为三大主流目标跟踪数据集（TNL2K、LaSOT、MGIT）提供标准化的无监督场景变化检测与关键帧提取方案。通过视觉大模型提取特征并计算帧间相似度，动态筛选不同比例的关键帧，为后续的 Tracking 评测提供高质量的基础数据。

---

## 💻 第一部分：代码与脚本仓库说明 (`/data/mhf/change_detect`)

本目录包含所有自动化提取脚本。脚本针对不同数据集的目录结构进行了深度适配，确保了提取过程的鲁棒性。

### 📁 核心脚本列表

| 脚本名称 | 对应数据集 | 特征提取模型 |
| --- | --- | --- |
| `batch_tnl2k_resnet_final.py` | TNL2K | ResNet50 |
| `batch_tnl2k_clip_final.py` | TNL2K | OpenCLIP (ViT-B/32) |
| `batch_lasot_resnet_final.py` | LaSOT | ResNet50 |
| `batch_lasot_open_clip.py` | LaSOT | OpenCLIP (ViT-B/32) |
| `batch_mgit_multiversion.py` | MGIT | ResNet50 |
| `batch_mgit_open_clip.py` | MGIT | OpenCLIP (ViT-B/32) |

### 🛠️ 脚本核心特性

1. **多版本并发生成 (Multi-version Generation)**
* **机制**：脚本配置 `PCT_VERSIONS = [10, 30, 50, 80]`。
* **优势**：单个序列仅需一次 GPU 推理，即可在内存中根据不同阈值生成 4 个版本的关键帧集合，大幅减少 I/O 损耗。


2. **数据集专属适配引擎**
* **TNL2K**：内置 **Fuzzy Match (模糊匹配)**。自动处理官方列表与实际文件夹名不一致（空格、大小写、后缀等）的问题。
* **LaSOT**：内置 **Category Penetration (类别穿透)**。支持 `Category/Sequence/img` 三级结构自动遍历。
* **MGIT**：内置 **Subfolder Auto-Discovery**。适配 `frame_xxx` 嵌套结构，并统一输出字段命名。


3. **坏数据免疫机制**
* 针对 TNL2K 等数据集中存在的损坏图像或 0 字节文件，脚本通过 `try...except` 结合预处理校验，自动跳过异常序列，确保产出的 JSON 数据 100% 可用。



---

## 📊 第二部分：产出数据仓库说明 (`/data/DATASETS_PUBLIC/SOIBench1`)

产出数据采用扁平化、语义化的目录结构，方便 Benchmark 脚本直接索引。

### 📁 目录树架构

```text
SOIBench1/
 ├── scene_changes_clip/              # OpenCLIP (ViT-B/32) 提取结果
 │   ├── top_10/                      # 阈值：保留变化最剧烈的前 10% 帧
 │   │   ├── lasot/ (train/test)
 │   │   ├── mgit/ (train/val/test)
 │   │   └── tnl2k/ (train/test)
 │   ├── top_30/
 │   ├── top_50/
 │   └── top_80/
 │
 └── scene_changes_resnet/            # ResNet50 提取结果
     ├── top_10/
     ├── top_30/
     └── ... (结构同上)

```

### 📄 标准化 JSON 格式规范

每个视频序列对应一个 JSON 文件，格式如下：

```json
{
    "sequence": "frame_001",              // 序列文件夹名
    "input_dir": "/data/.../frame_001",   // 图像原始路径
    "backend": "open_clip",               // 提取后端: "resnet50" 或 "open_clip"
    "model_name": "open_clip_ViT-B-32",   // 具体模型架构
    "index_base": 0,                      // 索引起始值
    "total_frames": 1500,                 // 总帧数
    "threshold": 0.0534,                  // 动态计算的余弦距离阈值
    "pct_top": 10,                        // 所属 Top X% 版本
    "key_frames": [0, 15, 23, 1499],      // 核心产出：关键帧索引列表 (含首尾)
    "num_key_frames": 4                   // 关键帧总数
}

```

---

## 🚀 使用与复现指南

### 1. 环境准备

* **基础依赖**：`torch`, `torchvision`, `Pillow`, `tqdm`
* **CLIP 支持**：`pip install open_clip_torch`

### 2. 权重配置

脚本默认从本地读取预训练权重，请确保以下路径存在对应文件：

* `/data/mhf/weights/resnet50-0676ba61.pth`
* `/data/mhf/weights/open_clip_pytorch_model.bin`

### 3. 修改配置并运行

1. 打开对应脚本（如 `batch_tnl2k_clip_final.py`）。
2. 定位至顶部 `# ============== 1. 配置区域 ==============`。
3. 根据需求修改 `SPLIT_NAME`（如 `"train"` 或 `"test"`）。
4. 执行运行指令：
```bash
python batch_tnl2k_clip_final.py

```
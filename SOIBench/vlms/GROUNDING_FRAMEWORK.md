# SOIBench VLM Grounding 推理框架

## 架构说明

本框架采用**适配器模式**，支持任意 VLM 模型的快速接入。

### 文件结构

```
SOIBench/vlms/
├── model_adapters/              # 模型适配器目录
│   ├── __init__.py             # 适配器注册表
│   ├── base.py                 # 适配器基类（定义接口）
│   ├── qwen3vl_adapter.py      # Qwen3VL 适配器
│   ├── glm46v_adapter.py       # GLM-4.6V 适配器
│   └── deepseekvl_adapter.py   # DeepSeek-VL2 适配器
├── grounding_common.py          # 通用函数和主流程
├── run_grounding.py             # 统一入口脚本
├── qwen3vl_infer.py            # Qwen3VL 推理引擎
├── glm46v_infer.py             # GLM-4.6V 推理引擎
└── deepseekvl_infer.py         # DeepSeek-VL2 推理引擎
```

## 使用方法

### 1. 使用现有模型

```bash
# Qwen3VL API 推理
python run_grounding.py --model qwen3vl --mode api

# GLM-4.6V 本地推理
python run_grounding.py --model glm46v --mode local

# DeepSeek-VL2 API 推理
python run_grounding.py --model deepseekvl --mode api --save_debug_vis
```

### 2. 添加新模型

只需三步即可接入新的 VLM 模型：

#### Step 1: 创建推理引擎（可选）

如果模型有特殊的推理逻辑，创建 `myvlm_infer.py`：

```python
# myvlm_infer.py
class MyVLMLocalEngine:
    def __init__(self, model_path):
        # 加载模型
        pass
    
    def chat(self, image_path, prompt, max_new_tokens=512):
        # 推理
        return response

class MyVLMAPIEngine:
    def __init__(self, api_key, model_name, ...):
        # 初始化 API 客户端
        pass
    
    def chat(self, image_path, prompt, max_new_tokens=512):
        # 调用 API
        return response

def parse_myvlm_bbox(response, img_width, img_height):
    # 解析 bbox
    return [[x1, y1, x2, y2], ...]
```

#### Step 2: 创建适配器

创建 `model_adapters/myvlm_adapter.py`：

```python
from typing import List
from .base import ModelAdapter

class MyVLMAdapter(ModelAdapter):
    """我的 VLM 模型适配器"""
    
    def build_prompt(self, desc_parts: List[str]) -> str:
        """构造模型特定的 prompt"""
        description = " ".join(desc_parts)
        return f"<my_special_format>{description}</my_special_format>"
    
    def parse_response(self, response: str, img_width: int, img_height: int) -> List[List[float]]:
        """解析模型输出的 bbox"""
        from myvlm_infer import parse_myvlm_bbox
        return parse_myvlm_bbox(response, img_width, img_height)
    
    def create_engine(self, args):
        """创建推理引擎"""
        if args.mode == 'local':
            from myvlm_infer import MyVLMLocalEngine
            return MyVLMLocalEngine(args.model_path)
        else:
            from myvlm_infer import MyVLMAPIEngine
            return MyVLMAPIEngine(
                api_key=args.api_key,
                model_name=args.api_model_name,
                ...
            )
    
    def get_default_model_path(self) -> str:
        return "/path/to/default/model"
    
    def get_default_api_model_name(self) -> str:
        return "my-vlm-model"
```

#### Step 3: 注册适配器

在 `model_adapters/__init__.py` 中注册：

```python
from .myvlm_adapter import MyVLMAdapter

ADAPTER_REGISTRY = {
    'qwen3vl': Qwen3VLAdapter,
    'glm46v': GLM46VAdapter,
    'deepseekvl': DeepSeekVLAdapter,
    'myvlm': MyVLMAdapter,  # 添加这一行
}
```

然后就可以使用了：

```bash
python run_grounding.py --model myvlm --mode api
```

## 适配器接口说明

所有适配器必须继承 `ModelAdapter` 并实现以下方法：

### 必须实现的方法

1. **`build_prompt(desc_parts: List[str]) -> str`**
   - 功能：构造模型特定的 prompt
   - 输入：处理后的描述文本列表（已添加标点）
   - 输出：模型特定格式的 prompt 字符串

2. **`parse_response(response: str, img_width: int, img_height: int) -> List[List[float]]`**
   - 功能：解析模型输出，提取 bbox
   - 输入：模型原始输出、图像宽高
   - 输出：bbox 列表，每个为 `[x1, y1, x2, y2]` 像素坐标

3. **`create_engine(args) -> Any`**
   - 功能：创建推理引擎
   - 输入：命令行参数对象
   - 输出：推理引擎对象（需有 `chat(image_path, prompt)` 方法）

### 可选重写的方法

1. **`get_default_model_path() -> str`**
   - 返回默认的本地模型路径

2. **`get_default_api_model_name() -> str`**
   - 返回默认的 API 模型名称

3. **`get_default_api_base_url() -> str`**
   - 返回默认的 API Base URL

4. **`preprocess_description(desc_parts: List[str]) -> List[str]`**
   - 预处理描述文本（可选）

5. **`postprocess_bboxes(bboxes: List[List[float]], img_width: int, img_height: int) -> List[List[float]]`**
   - 后处理 bbox（可选）

## 优势

✅ **统一接口**: 所有模型使用相同的命令行参数和流程
✅ **易于扩展**: 新增模型只需实现适配器，无需修改主流程
✅ **代码复用**: 通用函数（路径修复、断点续跑、可视化）只写一次
✅ **灵活配置**: 支持本地模型和 API 两种模式
✅ **类型安全**: 使用抽象基类定义清晰的接口

## 示例：完整的推理流程

```bash
# 1. Qwen3VL API 推理
export DASHSCOPE_API_KEY='your-key'
python run_grounding.py \
  --model qwen3vl \
  --mode api \
  --exp_tag qwen3vl_test \
  --save_debug_vis

# 2. GLM-4.6V 本地推理
python run_grounding.py \
  --model glm46v \
  --mode local \
  --model_path /path/to/glm46v \
  --exp_tag glm46v_local

# 3. DeepSeek-VL2 API 推理
export SILICONFLOW_API_KEY='your-key'
python run_grounding.py \
  --model deepseekvl \
  --mode api \
  --output_root ./results \
  --exp_tag deepseekvl_test
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

可以使用相同的评测脚本进行评测。

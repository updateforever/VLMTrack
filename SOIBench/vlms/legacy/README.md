# Legacy Scripts (已废弃)

本目录包含旧版的独立推理脚本，已被新的统一框架取代。

## 旧脚本列表

- `run_grounding_qwen3vl.py` - Qwen3VL 独立推理脚本
- `run_grounding_glm46v.py` - GLM-4.6V 独立推理脚本  
- `run_grounding_deepseekvl.py` - DeepSeek-VL2 独立推理脚本

## ⚠️ 不推荐使用

这些脚本已被新的适配器架构取代，不再维护。

**请使用新的统一脚本**: `../run_grounding.py`

## 新旧对比

### 旧方式（已废弃）
```bash
python run_grounding_qwen3vl.py --mode api --exp_tag test
python run_grounding_glm46v.py --mode local --model_path /path/to/model
python run_grounding_deepseekvl.py --mode api
```

### 新方式（推荐）
```bash
python run_grounding.py --model qwen3vl --mode api --exp_tag test
python run_grounding.py --model glm46v --mode local
python run_grounding.py --model deepseekvl --mode api
```

## 为什么废弃？

1. **代码重复**: 每个模型都有独立的主流程，维护成本高
2. **不易扩展**: 添加新模型需要复制整个脚本
3. **不一致**: 三个脚本的实现细节存在差异

## 新架构优势

1. ✅ **统一接口**: 所有模型使用相同的命令行和流程
2. ✅ **易于扩展**: 新增模型只需实现适配器
3. ✅ **代码复用**: 主流程只写一次
4. ✅ **易于维护**: 修改一次，所有模型受益

详见: `../GROUNDING_FRAMEWORK.md`

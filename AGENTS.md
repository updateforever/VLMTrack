# 仓库指南

## 项目定位
VLMTrack 不是普通的单目标跟踪仓库，而是一个围绕“认知跟踪”展开的研究平台。底座是 SUTrack 统一跟踪框架，研究重点是把 Qwen3-VL、GLM-4.6V、DeepSeek-VL 等视觉语言模型引入跟踪与文本引导 grounding，使系统具备全图搜索、语义记忆、状态推理和关键帧稀疏执行能力。贡献代码前，先明确你的改动属于哪条主线：`SUTrack 训练/评测`、`VLM 跟踪器`、`SOIBench grounding`，不要把三类逻辑混在一起。

## 代码结构与研究分层
`tracking/` 是训练、测试、结果分析和路径初始化入口。`lib/models/sutrack/`、`lib/train/` 对应 SUTrack 模型与训练框架；`lib/test/` 是当前研究主战场，其中 `tracker/` 放三类 VLM tracker 与 `vlm_engine.py`、`prompts.py`、`vlm_utils.py` 等共享模块，`parameter/` 放参数预设，`evaluation/` 放数据集与运行循环，`analysis/` 放结果分析。`SOIBench/vlms/` 是独立的 grounding 子系统，采用 adapter 模式接入多种 VLM；优先扩展 `model_adapters/` 与 `run_grounding.py`，不要继续往 `legacy/` 堆逻辑。

## 核心实验范式
当前主要维护三种 VLM 跟踪器：`qwen_vlm_visual`（两图/三图视觉匹配）、`qwen_vlm_cognitive`（语义记忆驱动的认知跟踪）、`qwen_vlm_hybrid`（视觉与认知混合触发）。如果你在做 prompt、记忆更新、状态输出、关键帧策略或 API/本地模型切换，默认修改点在 `lib/test/tracker/` 与 `lib/test/parameter/vlm_common.py`。如果你在做 grounding 实验，入口是：
```bash
python SOIBench/vlms/run_grounding.py --model qwen3vl --mode api
python SOIBench/vlms/eval_results.py --pred_root ./results
```

## 常用命令
```bash
conda create -n sutrack python=3.8
conda activate sutrack
bash install.sh
export PYTHONPATH=$(pwd):$PYTHONPATH
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
python tracking/train.py --script sutrack --config sutrack_b224 --save_dir . --mode single
python tracking/test.py qwen_vlm_cognitive default --dataset_name lasot --debug 1 --threads 1
python tracking/test.py qwen_vlm_hybrid default_visual_kf --dataset_name lasot --threads 2
bash test_cognitive_tracking.sh
```
多卡训练仍走 `lib/train/run_training.py`；API 模式需配置 `DASHSCOPE_API_KEY`，部分 grounding 模型还依赖 `SILICONFLOW_API_KEY`。

## 开发约束
Python 统一 4 空格缩进，函数/模块使用 `snake_case`，类使用 `PascalCase`。新增 tracker 时，至少同步检查三层是否一致：`lib/test/tracker/<name>.py`、`lib/test/parameter/<name>.py`、对应 prompt 或公共参数。Prompt 模板集中放在 `lib/test/tracker/prompts.py`，不要把长 prompt 重新硬编码回 tracker。关键帧逻辑放 `lib/test/evaluation/keyframe_loader.py` 一侧，避免在 tracker 内重复实现。路径、数据集根目录和本机配置放 `lib/test/evaluation/local.py`，不要把绝对路径写死进可提交代码。

## 测试与结果要求
这个仓库主要靠 benchmark run 验证，不靠单元测试。提交前至少做一次单序列或 `--debug 1` 快速验证，再跑一个代表性数据集。涉及 bbox 解析、状态输出或 NaN 填充时，必须检查结果文件是否与现有评测脚本兼容。跟踪结果通常写入 `results/{tracker}/{dataset}/`，grounding 结果写入 `SOIBench/vlms/results/` 或你指定的 `output_root`；不要把大规模实验结果、可视化产物或本地路径文件提交进仓库。

## 提交与 PR 规范
近期历史里有不少 `add` 这类低信息提交，不建议继续沿用。请使用明确的祈使句，如 `feat: add cognitive tracker state parser`、`fix: correct keyframe dataset mapping`、`docs: update grounding workflow`。PR 需要写清研究目的、影响模块、运行命令、使用的数据集、关键指标或可视化变化；若改动影响认知跟踪范式或 grounding 输出格式，必须说明兼容性影响。

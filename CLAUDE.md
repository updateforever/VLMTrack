# CLAUDE.md

本文件记录当前仓库在 2026-04 的真实工作流，重点覆盖 `vlm_cognitive_mosaic`、本地 vLLM 部署、`tracker_param` 语法和最近一次重构后的约定。

## 当前重点

当前主线不是传统 SUTrack 训练，而是围绕 `vlm_cognitive_mosaic` 做认知跟踪实验：

- 使用 `tracker_name` 控制跟踪范式
- 使用 `tracker_param` 控制部署方式、模型版本、prompt 版本和 mosaic 额外参数
- 使用本地 vLLM API 优先替代本地 `transformers` 直载大模型

当前最常用的范式是：

```bash
python tracking/test.py vlm_cognitive_mosaic <tracker_param> ...
```

## 最新进展

最近已经完成或确认的改动：

- 跟踪器命名统一为：
  - `vlm_cognitive_mosaic`
  - `vlm_cognitive`
  - `vlm_visual`
  - `vlm_hybrid`
- 旧的 `qwen_*` tracker_name 约定已不再作为主入口使用
- `tracker_param` 已收敛为“部署方式 + 模型 + 后缀”的单参数体系
- `vlm_cognitive_mosaic` 已支持 `prompt_v2` 实验，通过 `tracker_param` 的 `_v2` 后缀控制
- `vlm_cognitive_mosaic` 已支持 `_ref` 坐标锚定 prompt
- `vlm_cognitive_mosaic` 已统一使用单字段 `cognition_chain`
- 初始化语义记忆已转向“初始认知构建”思路，不再维护老的 appearance / motion / context 三字段
- API 模式已支持本地 vLLM，通过 `LOCAL_VLLM_BASE_URL` 和 `LOCAL_VLLM_API_KEY` 指向不同端口
- 已补充本地 vLLM alias：
  - `api_vllm_qwen25_vl_32b`
  - `api_vllm_qwen25_vl_7b`
  - `api_vllm_qwen3_vl_4b_thinking`
  - `api_vllm_qwen35_9b`

## tracker_name 与 tracker_param

### tracker_name

`tracker_name` 只负责指定“跟踪范式”，不要再把模型信息塞进这里。

当前主要使用：

- `vlm_cognitive_mosaic`
- `vlm_cognitive`
- `vlm_visual`
- `vlm_hybrid`

### tracker_param

`tracker_param` 负责控制：

- 部署方式：`api_*` 或 `local_*`
- 具体模型：例如 `vllm_qwen25_vl_32b`
- mosaic 额外参数：`_bN` / `_sM` / `_ref` / `_v2`

对 `vlm_cognitive_mosaic`，当前语法为：

```text
<deploy_and_model>[_bN][_sM][_ref][_v2]
```

示例：

- `api_vllm_qwen25_vl_32b`
- `api_vllm_qwen25_vl_32b_v2`
- `api_vllm_qwen25_vl_32b_ref_v2`
- `api_vllm_qwen25_vl_7b_b5_s15_v2`
- `local_qwen3vl_4b_thinking_ref`

后缀含义：

- `_bN`：`history_buffer_size = N`
- `_sM`：`sample_interval = M`
- `_ref`：使用带初始 bbox 坐标锚点的 mosaic prompt
- `_v2`：使用 `prompt_v2`，即新的自由启发式认知推理 prompt

注意：

- `_v2` 是直接挂在 `tracker_param` 末尾的
- `_v2` 可以和 `_ref`、`_bN`、`_sM` 组合
- 非法 `tracker_param` 现在会直接报错，不再 silently fallback 到 `default`

## 本地 vLLM API 约定

### 核心机制

本地 vLLM API 模式不需要改代码中的端口常量，直接按命令传环境变量：

```bash
LOCAL_VLLM_BASE_URL=http://127.0.0.1:8000/v1
LOCAL_VLLM_API_KEY=local-test-key
```

`vlm_common.py` 会在 `api_vllm_*` 这类 alias 下自动读取这两个环境变量。

因此：

- 端口适配不靠改代码
- 端口适配靠每条命令前显式设置 `LOCAL_VLLM_BASE_URL`
- 这样可以同时对不同端口起不同模型，只要每个推理任务在自己的 shell 命令里指定对应端口

### 当前推荐 alias

- `api_vllm_qwen25_vl_32b`
- `api_vllm_qwen25_vl_7b`
- `api_vllm_qwen3_vl_4b_thinking`
- `api_vllm_qwen35_9b`

## 当前可用模型情况

本地目录当前已确认存在：

- `/root/user-data/MODEL_WEIGHTS_PUBLIC/MLLM_weights/Qwen2_5-VL-32B-Instruct`
- `/root/user-data/MODEL_WEIGHTS_PUBLIC/MLLM_weights/Qwen2_5-VL-7B-Instruct`
- `/root/user-data/MODEL_WEIGHTS_PUBLIC/MLLM_weights/Qwen3-VL-4B-Thinking`
- `/root/user-data/MODEL_WEIGHTS_PUBLIC/MLLM_weights/Qwen3_5-9B`

实测结论：

- `Qwen2.5-VL-32B-Instruct`：可通过 vLLM 正常部署
- `Qwen2.5-VL-7B-Instruct`：已准备对应部署脚本，适合作为更轻量的本地 API 模型
- `Qwen3-VL-4B-Thinking`：本地已有权重，可作为 Qwen3-VL 路线的现成实验模型
- `Qwen3_5-9B`：当前 vLLM 环境暂时无法部署，原因是 `transformers` 对 `qwen3_5` 架构不识别

## vLLM 部署脚本

当前仓库里已经有几个极简部署脚本：

- [scripts/start_vllm_qwen25_vl_32b.sh](/root/user-data/wyp/VLMTrack/scripts/start_vllm_qwen25_vl_32b.sh)
- [scripts/start_vllm_qwen35_9b.sh](/root/user-data/wyp/VLMTrack/scripts/start_vllm_qwen35_9b.sh)
- [scripts/start_vllm_qwen25_vl_7b_gpu2.sh](/root/user-data/wyp/VLMTrack/scripts/start_vllm_qwen25_vl_7b_gpu2.sh)

这些脚本风格统一：

- 只保留参数区
- 启动命令走 `nohup`
- 输出最基本的 `pid` 和 `log`

## 两条常用 Mosaic 推理命令

下面这两条是当前最推荐直接使用的命令。它们显式按端口指定本地 vLLM 地址，同时带上 `_v2` 后缀，方便测试新的 prompt。

### 1. Qwen2.5-VL-32B

假设 `Qwen2.5-VL-32B` 部署在 `8000`：

```bash
LOCAL_VLLM_BASE_URL=http://127.0.0.1:8000/v1 \
LOCAL_VLLM_API_KEY=local-test-key \
python tracking/test.py \
  vlm_cognitive_mosaic \
  api_vllm_qwen25_vl_32b_v2 \
  --dataset_name videocube_val_tiny \
  --threads 0 \
  --debug 2 \
  --sequence 029
```

### 2. Qwen2.5-VL-7B

假设 `Qwen2.5-VL-7B` 部署在 `8002`：

```bash
LOCAL_VLLM_BASE_URL=http://127.0.0.1:8002/v1 \
LOCAL_VLLM_API_KEY=local-test-key \
python tracking/test.py \
  vlm_cognitive_mosaic \
  api_vllm_qwen25_vl_7b_v2 \
  --dataset_name videocube_val_tiny \
  --threads 0 \
  --debug 2 \
  --sequence 029
```

如果想同时测试坐标锚定和 prompt_v2，可以把 `tracker_param` 改成：

```text
api_vllm_qwen25_vl_32b_ref_v2
api_vllm_qwen25_vl_7b_ref_v2
```

如果想改 mosaic 历史缓冲参数，例如：

```text
api_vllm_qwen25_vl_32b_b5_s15_v2
api_vllm_qwen25_vl_7b_b5_s15_ref_v2
```

## 调试建议

当前调试 `vlm_cognitive_mosaic` 时，推荐拆成两部分：

### 1. 一个终端负责模型服务

例如：

```bash
bash scripts/start_vllm_qwen25_vl_32b.sh
```

看服务日志：

```bash
tail -f /workspace/tmp/vllm_qwen25_vl_32b.log
```

### 2. 另一个终端负责跟踪调试

例如：

```bash
LOCAL_VLLM_BASE_URL=http://127.0.0.1:8000/v1 \
LOCAL_VLLM_API_KEY=local-test-key \
python tracking/test.py \
  vlm_cognitive_mosaic \
  api_vllm_qwen25_vl_32b_v2 \
  --dataset_name videocube_val_tiny \
  --threads 0 \
  --debug 2 \
  --sequence 029
```

重点断点位置：

- `tracking/test.py`
- `lib/test/evaluation/tracker.py`
- `lib/test/tracker/vlm_cognitive_mosaic.py`
- `lib/test/tracker/vlm_engine.py`
- `lib/test/tracker/vlm_utils.py`

## Prompt 体系

当前 `vlm_cognitive_mosaic` 相关 prompt 至少有两组主变体：

- 常规版本：
  - `cognitive_mosaic`
  - `cognitive_mosaic_ref`
- `prompt_v2` 实验版本：
  - `cognitive_mosaic_v2`
  - `cognitive_mosaic_ref_v2`

初始化 prompt 当前沿用：

- `init_story_mosaic`

当前认知输出统一围绕：

- `cognition_chain`

不再继续维护旧版三字段：

- `appearance`
- `motion`
- `context`

## 代码结构中的关键位置

### 跟踪入口

- [tracking/test.py](/root/user-data/wyp/VLMTrack/tracking/test.py)

### Mosaic 跟踪器

- [lib/test/tracker/vlm_cognitive_mosaic.py](/root/user-data/wyp/VLMTrack/lib/test/tracker/vlm_cognitive_mosaic.py)
- [lib/test/parameter/vlm_cognitive_mosaic.py](/root/user-data/wyp/VLMTrack/lib/test/parameter/vlm_cognitive_mosaic.py)

### VLM 推理

- [lib/test/tracker/vlm_engine.py](/root/user-data/wyp/VLMTrack/lib/test/tracker/vlm_engine.py)
- [lib/test/parameter/vlm_common.py](/root/user-data/wyp/VLMTrack/lib/test/parameter/vlm_common.py)

### Prompt 与解析

- [lib/test/tracker/prompts.py](/root/user-data/wyp/VLMTrack/lib/test/tracker/prompts.py)
- [lib/test/tracker/vlm_utils.py](/root/user-data/wyp/VLMTrack/lib/test/tracker/vlm_utils.py)

## 需要特别注意的坑

### 1. vLLM 与本地环境污染

之前排查过 `PYTHONPATH / PIP_TARGET / PYTHONUSERBASE` 污染问题。  
凡是部署本地 vLLM，优先使用干净环境，必要时显式：

```bash
unset PYTHONPATH PIP_TARGET PYTHONUSERBASE
```

### 2. `debug=2` 会触发更多输出

`vlm_cognitive_mosaic` 在 `debug=2` 下更容易触发可视化与详细日志保存。  
如果只想轻量跑推理，不要默认带 `--debug 2`。

### 3. JSON 解析不能盲信模型

模型经常输出：

- JSON 主体
- 额外 markdown fence
- 额外 explanation

因此 `vlm_utils.py` 的解析必须按“抽取首个 JSON 对象”思路做鲁棒处理，不能直接假设整个字符串都能 `json.loads(...)`。

### 4. Qwen3.5-9B 当前不是 prompt 问题，是部署问题

当前 `Qwen3_5-9B` 起不来，不是 tracker 或 prompt 错，而是：

- `vllm` 环境里的 `transformers` 暂不识别 `qwen3_5`
- 国内镜像升级可作为后续尝试
- 现阶段不要把时间浪费在 tracker 逻辑上

## 当前建议

当前最务实的实验路线：

1. 主力模型使用 `Qwen2.5-VL-32B-Instruct`
2. 对比模型使用 `Qwen2.5-VL-7B-Instruct`
3. `tracker_param` 统一带 `_v2`，先完成新 prompt 的横向比较
4. 如需更小的 Qwen3 系列，优先尝试 `Qwen3-VL-4B-Thinking` 或后续下载 `Qwen3-VL-8B-Instruct`

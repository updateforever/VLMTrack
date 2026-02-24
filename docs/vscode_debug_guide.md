# VSCode Debug配置说明

## 📝 概述

VSCode的debug配置已更新，完全适配新的tracker和参数体系。

## 🎯 可用的Debug配置

### VLMTrack Tracker配置（9个）

#### 基础两图跟踪 (qwen_vlm)
1. **VLMTrack: qwen_vlm (基础两图) - API**
   - Tracker: `qwen_vlm`
   - Config: `api`
   - Dataset: LaSOT
   - Sequence: airplane-1
   - Debug Level: 2 (保存可视化)

2. **VLMTrack: qwen_vlm (基础两图) - Local 4B**
   - Tracker: `qwen_vlm`
   - Config: `local_4b`
   - Dataset: LaSOT
   - Sequence: airplane-1

#### 三图跟踪 (qwen_vlm_three)
3. **VLMTrack: qwen_vlm_three (三图跟踪) - API**
   - Tracker: `qwen_vlm_three`
   - Config: `api`
   - Dataset: LaSOT
   - Sequence: bear-1

4. **VLMTrack: qwen_vlm_three (三图跟踪) - Local 4B**
   - Tracker: `qwen_vlm_three`
   - Config: `local_4b`
   - Dataset: LaSOT
   - Sequence: bear-1

#### 记忆库跟踪 (qwen_vlm_memory)
5. **VLMTrack: qwen_vlm_memory (记忆库) - API**
   - Tracker: `qwen_vlm_memory`
   - Config: `api`
   - Dataset: LaSOT
   - Sequence: bird-1

6. **VLMTrack: qwen_vlm_memory (记忆库) - Local 8B**
   - Tracker: `qwen_vlm_memory`
   - Config: `local_8b`
   - Dataset: LaSOT
   - Sequence: bird-1

#### TNL2K数据集测试
7. **VLMTrack: TNL2K - qwen_vlm**
8. **VLMTrack: TNL2K - qwen_vlm_three**
9. **VLMTrack: TNL2K - qwen_vlm_memory**

### SOIBench配置（5个）
- SOIBench: Qwen3VL API 推理
- SOIBench: GLM-4.6V 推理 (API)
- SOIBench: deepseekV 推理 (API)
- SOIBench: 评测结果 (含人类基线)
- SOIBench: 可视化结果 (含人类基线)

---

## 🚀 使用方法

### 1. 选择Debug配置

在VSCode中：
1. 按 `F5` 或点击左侧的"运行和调试"图标
2. 在顶部下拉菜单中选择要运行的配置
3. 点击绿色播放按钮或按 `F5` 开始调试

### 2. 设置断点

在代码行号左侧点击设置断点，程序会在断点处暂停。

### 3. Debug级别说明

| Debug Level | 说明 |
|-------------|------|
| 0 | 无输出 |
| 1 | 打印关键信息到控制台 |
| 2 | 保存可视化图片到results目录 |
| 3 | 保存图片 + 实时显示窗口 |

---

## 📊 配置结构

### 基本格式

```json
{
    "name": "VLMTrack: qwen_vlm (基础两图) - API",
    "type": "debugpy",
    "request": "launch",
    "program": "${workspaceFolder}/tracking/test.py",
    "console": "integratedTerminal",
    "args": [
        "qwen_vlm",      // tracker_name
        "api",           // tracker_param
        "--dataset_name", "lasot",
        "--sequence", "airplane-1",
        "--debug", "2"
    ],
    "env": {
        "PYTHONPATH": "${workspaceFolder}",
        "DASHSCOPE_API_KEY": "your_api_key"
    }
}
```

### 关键参数说明

| 参数 | 说明 |
|------|------|
| `tracker_name` | Tracker名称 (qwen_vlm, qwen_vlm_three, qwen_vlm_memory) |
| `tracker_param` | 配置名称 (api, local_4b, local_8b) |
| `--dataset_name` | 数据集 (lasot, tnl2k, videocube) |
| `--sequence` | 序列名或序号 |
| `--debug` | Debug级别 (0-3) |
| `--threads` | 线程数 (0=单线程调试) |

---

## 🔧 自定义配置

### 修改测试序列

在配置中修改 `--sequence` 参数：

```json
"args": [
    "qwen_vlm",
    "api",
    "--dataset_name", "lasot",
    "--sequence", "car-1",  // 修改这里
    "--debug", "2"
]
```

### 切换数据集

修改 `--dataset_name` 参数：

```json
"args": [
    "qwen_vlm",
    "api",
    "--dataset_name", "tnl2k",  // 切换到TNL2K
    "--sequence", "0",
    "--debug", "1"
]
```

### 使用本地模型

将 `tracker_param` 改为 `local_4b` 或 `local_8b`：

```json
"args": [
    "qwen_vlm",
    "local_4b",  // 使用本地4B模型
    "--dataset_name", "lasot",
    "--sequence", "airplane-1"
]
```

并移除 `DASHSCOPE_API_KEY` 环境变量。

---

## 🎨 Debug工作流

### 典型调试流程

1. **选择配置**
   - 根据需要选择API或Local模式
   - 选择合适的tracker版本

2. **设置断点**
   - 在 `qwen_vlm.py` 的 `track()` 方法设置断点
   - 在 `vlm_engine.py` 的 `infer()` 方法设置断点

3. **启动调试**
   - 按 `F5` 开始
   - 程序会在断点处暂停

4. **检查变量**
   - 查看 `prompt` 内容
   - 查看 `output` 结果
   - 检查 `bbox` 解析

5. **单步执行**
   - `F10`: 单步跳过
   - `F11`: 单步进入
   - `Shift+F11`: 单步跳出

---

## 📝 常见调试场景

### 场景1: 调试Prompt生成

设置断点位置：
```python
# lib/test/tracker/qwen_vlm.py
def track(self, image, info: dict = None):
    # 设置断点在这里 👇
    prompt = get_prompt(self.prompt_name, ...)
```

### 场景2: 调试VLM推理

设置断点位置：
```python
# lib/test/tracker/vlm_engine.py
def infer(self, images: List[np.ndarray], prompt: str):
    # 设置断点在这里 👇
    if self.mode == 'api':
        return self._infer_api(images, prompt)
```

### 场景3: 调试Bbox解析

设置断点位置：
```python
# lib/test/tracker/qwen_vlm.py
def track(self, image, info: dict = None):
    output = self.vlm.infer([template_with_box, image], prompt)
    # 设置断点在这里 👇
    bbox_xyxy = parse_bbox_from_text(output, W, H)
```

---

## 💡 Debug技巧

### 1. 使用条件断点

右键断点 → 编辑断点 → 添加条件：
```python
self.frame_id == 10  # 只在第10帧暂停
```

### 2. 查看Debug Console

在Debug Console中可以执行Python代码：
```python
>>> print(prompt)  # 打印prompt内容
>>> len(images)    # 查看图像数量
```

### 3. 修改变量值

在调试时可以在Variables面板修改变量值，测试不同情况。

### 4. 日志断点

右键断点 → 编辑断点 → Log Message：
```
Frame {self.frame_id}: Output length = {len(output)}
```

这样可以在不停止程序的情况下输出信息。

---

## 🔑 环境变量

### API密钥配置

在配置中设置：
```json
"env": {
    "DASHSCOPE_API_KEY": "your_api_key_here"
}
```

或在系统环境变量中设置：
```bash
export DASHSCOPE_API_KEY="your_api_key"
```

---

## ✨ 总结

- ✅ 9个VLMTrack配置，覆盖所有tracker和模式
- ✅ API和Local模式都包含
- ✅ 不同数据集配置
- ✅ 完全适配新的框架规范

**VSCode debug配置已完善，可以方便地调试各种情况！** 🚀

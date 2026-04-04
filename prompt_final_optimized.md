# 认知跟踪 Prompt（最终优化版 - 中文）

```
# === 任务：认知视觉跟踪 ===

你正在执行认知视觉跟踪 - 在视频帧序列中保持对目标物的持续感知。

## 输入说明

**图1（历史参考拼接图）**包含：
- **初始模板**：帧 #0，绿色边界框标出目标（真值标注）
- **历史轨迹参考**：若干历史帧，红色边界框为之前的预测结果（可能有误）

**图2（当前帧）**：需要你在此定位目标

**长期记忆（叙事）**：
{memory_story}

**初始目标描述**（可选）：
{language_description}

## 你的目标
1. 基于给定信息识别目标
2. 推理目标的状态和位置，并给出证据
3. 更新长期记忆以促进后续跟踪

---

# === 输出要求 ===

## 1. 当前帧预测

### 目标状态（选择一个选项）：
A. normal - 目标清晰可见
B. partially_occluded - 目标部分被遮挡但可识别
C. fully_occluded - 目标完全被遮挡但可能仍在场景中
D. out_of_view - 目标移出画面边界
E. disappeared - 目标从场景中消失
F. reappeared - 目标在消失后重新出现

### 边界框：
- 如果可见或可推测位置（A/B/C/F）：提供 [x1, y1, x2, y2]，范围 0-1000
- 如果完全无法定位（D/E）：输出 [0, 0, 0, 0]

### 环境状态（选择所有适用的选项）：
A. normal - 正常
B. low_light - 低光照
C. high_light - 强光/过曝
D. motion_blur - 运动模糊
E. scene_change - 场景切换
F. viewpoint_change - 视角变化
G. scale_change - 尺度变化
H. crowded - 拥挤/相似物体多
I. background_clutter - 背景杂乱

## 2. 跟踪证据（短时记忆）
解释你对这一帧预测的推理（2-4句话）：
- 目标是什么，正在做什么？
- 为什么你相信这是（或不是）目标？
- 什么证据支持你的状态判断？

## 3. 置信度分数
你对你的预测的把握度 (0.0-1.0，0.1刻度)

## 4. 长期记忆更新（叙事）
更新目标旅程的故事（简洁但完整）：
- 描述目标的外观、运动轨迹、状态变化
- 保持叙事连贯性
- 包含对未来可能发展的预测

---

# === 输出格式 ===

{
  "target_status": "A",
  "environment_status": ["A"],
  "bbox": [x1, y1, x2, y2],
  "tracking_evidence": "The target is a red sedan moving right at moderate speed. The color, shape, and motion pattern match the initial template. Currently clearly visible with no occlusion.",
  "confidence": 0.9,
  "memory_update": {
    "story": "A red sedan with white side stripes traveling on an urban road. Started from left, moving steadily right. Maintained clear visibility throughout. Likely to continue rightward motion and may exit frame from right edge soon."
  }
}
```

---

# 英文版本

```
# === TASK: Cognitive Visual Tracking ===

You are performing cognitive visual tracking - maintaining continuous awareness of a target object across video frames.

## Input Description

**Image 1 (Historical Reference Mosaic)** contains:
- **Initial Template**: Frame #0 with GREEN bounding box (ground truth annotation)
- **Historical Trajectory Reference**: Several historical frames with RED bounding boxes (predicted results, may contain errors)

**Image 2 (Current Frame)**: Where you need to locate the target

**Long-term Memory (Narrative)**:
{memory_story}

**Initial Target Description** (optional):
{language_description}

## Your Goal
1. Identify the target based on given information
2. Reason about target's state and location with evidence
3. Update long-term memory to facilitate future tracking

---

# === OUTPUT REQUIREMENTS ===

## 1. Current Frame Prediction

### Target Status (choose ONE option):
A. normal - Target clearly visible
B. partially_occluded - Target partially blocked but identifiable
C. fully_occluded - Target completely blocked but likely still in scene
D. out_of_view - Target moved outside frame boundaries
E. disappeared - Target vanished from scene
F. reappeared - Target returned after being absent

### Bounding Box:
- If visible or location inferable (A/B/C/F): Provide [x1, y1, x2, y2] in 0-1000 scale
- If completely unlocatable (D/E): Output [0, 0, 0, 0]

### Environment Status (select ALL applicable options):
A. normal
B. low_light
C. high_light
D. motion_blur
E. scene_change
F. viewpoint_change
G. scale_change
H. crowded
I. background_clutter

## 2. Tracking Evidence (Short-term Memory)
Explain your reasoning for this frame's prediction (2-4 sentences):
- What is the target and what is it doing?
- Why do you believe this is (or isn't) the target?
- What evidence supports your status judgment?

## 3. Confidence Score
Your confidence in the prediction (0.0-1.0, 0.1 granularity)

## 4. Long-term Memory Update (Narrative)
Update the story of target's journey (concise but complete):
- Describe target's appearance, motion trajectory, state changes
- Maintain narrative coherence
- Include predictions about future developments

---

# === OUTPUT FORMAT ===

{
  "target_status": "A",
  "environment_status": ["A"],
  "bbox": [x1, y1, x2, y2],
  "tracking_evidence": "The target is a red sedan moving right at moderate speed. The color, shape, and motion pattern match the initial template. Currently clearly visible with no occlusion.",
  "confidence": 0.9,
  "memory_update": {
    "story": "A red sedan with white side stripes traveling on an urban road. Started from left, moving steadily right. Maintained clear visibility throughout. Likely to continue rightward motion and may exit frame from right edge soon."
  }
}
```

---

## 主要改进

| 改进点 | 修改内容 | 优势 |
|--------|---------|------|
| **图1说明** | ✅ 添加"输入说明"部分 | 结构更清晰 |
| **历史帧编号** | ✅ 改为"若干历史帧" | 更通用，不受更新影响 |
| **重要提示** | ✅ 删除 | 更简洁 |
| **状态选项** | ✅ 改为 A/B/C/D/E/F | 节约输出长度 |
| **环境选项** | ✅ 改为 A-I | 节约输出长度 |
| **全遮挡坐标** | ✅ C 可以输出坐标 | 更合理 |
| **故事要求** | ✅ 包含未来预测 | 更有前瞻性 |
| **格式要求** | ✅ 删除 | 更简洁 |

## 输出长度对比

**旧版本**：
```json
"target_status": "partially_occluded",
"environment_status": ["low_light", "motion_blur"]
```
约 70 字符

**新版本**：
```json
"target_status": "B",
"environment_status": ["B", "D"]
```
约 50 字符

**节约约 30% 输出长度** ✅

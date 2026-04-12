"""
VLM Tracker Prompt配置管理模块

统一管理所有tracker的prompt，支持通过配置名称读取不同的prompt模板。

设计理念:
  1. 任务分解: 明确子任务步骤，引导VLM chain-of-thought推理
  2. 反例引导: 明确告知哪些情况需要输出[0,0,0,0]
  3. 鲁棒性:   处理遮挡、出视野、相似物体干扰、外观变化等
  4. 格式严格: 要求仅输出JSON，减少解析失败
"""


class PromptTemplate:
    """Prompt模板基类"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def build(self, **kwargs) -> str:
        """构建prompt，子类需实现"""
        raise NotImplementedError


# ==================== 两图跟踪Prompt ====================

class TwoImageTrackingPrompt(PromptTemplate):
    """
    两图跟踪: 模板帧(绿框) + 当前帧
    改进: 引导关注discriminative特征、区分相似物体、明确遮挡处理
    """
    def __init__(self):
        super().__init__(
            name="two_image_tracking",
            description="Template image with bbox + current image"
        )

    def build(self, target_description: str = "the target object") -> str:
        return (
            "# --- TASK ---\n"
            "You are a precise visual object tracker. Locate the SAME target instance in the current frame.\n\n"

            "# --- INPUTS ---\n"
            f"Image 1 (Template): Target marked by GREEN box. Target: {target_description}.\n"
            "Image 2 (Current Frame): Search for the SAME object instance here.\n\n"

            "# --- TRACKING STRATEGY ---\n"
            "Step 1 - Study target in Image 1:\n"
            "  - Primary and secondary colors\n"
            "  - Shape, aspect ratio, and distinctive textures\n"
            "  - Any unique markings that distinguish it from similar objects\n\n"
            "Step 2 - Locate target in Image 2:\n"
            "  - Target may have moved, changed scale, or be partially occluded\n"
            "  - If multiple similar objects exist, pick the one with most matching features\n"
            "  - Consider position continuity (target unlikely to teleport)\n\n"
            "Step 3 - Handle edge cases:\n"
            "  - Fully occluded or out of frame → output [0,0,0,0]\n"
            "  - Severely truncated at image edge → include only visible part\n\n"

            "# --- OUTPUT FORMAT ---\n"
            "Respond with ONLY this JSON (no explanation outside JSON):\n"
            "{\n"
            '  "bbox": [x1, y1, x2, y2],  // 0-1000 scale. [0,0,0,0] if not visible.\n'
            '  "evidence": "Key matched features and spatial reasoning (1-2 sentences)",\n'
            '  "confidence": 0.9           // 0.0=lost, 1.0=certain\n'
            "}\n"
        )


# ==================== 三图跟踪Prompt ====================

class ThreeImageTrackingPrompt(PromptTemplate):
    """
    三图跟踪: 初始帧(绿框) + 上一帧(蓝框) + 当前帧
    改进: 更精确的运动预测引导，强调appearance from Image1 + motion from Image2
    """
    def __init__(self):
        super().__init__(
            name="three_image_tracking",
            description="Initial frame + previous frame + current frame"
        )

    def build(self, target_description: str = "the target object") -> str:
        return (
            "# --- TASK ---\n"
            "Track a target object using its initial appearance and recent motion cues.\n\n"

            "# --- INPUTS ---\n"
            f"Image 1 (Initial - GREEN box): Ground truth target appearance. Target: {target_description}.\n"
            "Image 2 (Previous - BLUE box): Last prediction. Use ONLY for motion direction estimation.\n"
            "Image 3 (Current Frame): Locate the target here.\n\n"

            "# --- TRACKING STRATEGY ---\n"
            "Step 1 - Learn appearance from Image 1 (authoritative reference):\n"
            "  - Colors, shape, texture, distinctive features\n"
            "  - This defines WHAT the target looks like\n\n"
            "Step 2 - Estimate motion from Image 1 → Image 2:\n"
            "  - Direction: which way is the target moving?\n"
            "  - Speed: how much did it move?\n"
            "  - Predict approximate location in Image 3\n\n"
            "Step 3 - Search in Image 3:\n"
            "  - Start near predicted location, search broader area if not found\n"
            "  - Appearance similarity to Image 1 takes priority over position\n"
            "  - Target may have changed scale or be partially occluded\n\n"
            "Step 4 - Edge cases:\n"
            "  - Not visible or fully occluded → output [0,0,0,0]\n\n"

            "# --- OUTPUT FORMAT ---\n"
            "Respond with ONLY this JSON:\n"
            "{\n"
            '  "bbox": [x1, y1, x2, y2],  // 0-1000 scale. [0,0,0,0] if not visible.\n'
            '  "evidence": "Appearance match from Image 1 + motion prediction from Image 2 (1-2 sentences)",\n'
            '  "confidence": 0.9           // 0.0=lost, 1.0=certain\n'
            "}\n"
        )


# ==================== 认知跟踪Prompt（新版）====================

class CognitiveTrackingPrompt(PromptTemplate):
    """
    认知跟踪: 语义记忆 + 全图搜索 + 结构化状态判断
    核心改进:
      1. 强制全图搜索，不依赖位置先验
      2. 结构化输出：target_status（选择题）+ environment_status（多选）
      3. 认知推理：tracking_evidence（主观性文本描述）
    """
    def __init__(self):
        super().__init__(
            name="cognitive_tracking",
            description="Cognitive tracking with global search and structured reasoning"
        )

    def build(self,
              memory_appearance: str = "",
              memory_motion: str = "",
              memory_context: str = "") -> str:
        return (
            "# --- TASK ---\n"
            "You are a cognitive tracker. Track the target using semantic memory and visual reasoning.\n"
            "DO NOT rely on previous position - search the ENTIRE image.\n\n"

            "# --- SEMANTIC MEMORY ---\n"
            f"Appearance: {memory_appearance}\n"
            f"Motion: {memory_motion}\n"
            f"Context: {memory_context}\n\n"

            "# --- INPUT ---\n"
            "Image 1 (Previous Frame - BLUE box): Last known position. Use ONLY for motion estimation, NOT as search constraint.\n"
            "Image 2 (Current Frame): Search the ENTIRE image for the target.\n\n"

            "# --- COGNITIVE TRACKING STRATEGY ---\n\n"
            "Step 1 - Analyze Current Scene:\n"
            "  - Is this the same scene as memory context?\n"
            "  - If scene changed significantly → mark 'scene_change' in environment_status\n"
            "  - Check lighting, blur, clutter, crowding\n\n"

            "Step 2 - Global Search (CRITICAL):\n"
            "  - Scan the ENTIRE image, not just near BLUE box\n"
            "  - Look for objects matching appearance memory\n"
            "  - Consider scale changes, rotation, partial occlusion\n"
            "  - Do NOT assume target is near previous position\n\n"

            "Step 3 - Target Status Judgment:\n"
            "  - normal: Target clearly visible, no occlusion\n"
            "  - partially_occluded: Target partially blocked but identifiable\n"
            "  - fully_occluded: Target completely blocked but likely still in scene\n"
            "  - out_of_view: Target moved out of frame\n"
            "  - disappeared: Target vanished from scene (entered building, etc.)\n"
            "  - reappeared: Target was absent but now visible again\n\n"

            "Step 4 - Environment Analysis:\n"
            "  Select ALL applicable conditions:\n"
            "  - normal: Normal lighting and scene\n"
            "  - low_light: Dark environment, low visibility\n"
            "  - high_light: Overexposure, backlight, glare\n"
            "  - motion_blur: Fast motion causing blur\n"
            "  - scene_change: Scene changed significantly (cut, viewpoint shift)\n"
            "  - viewpoint_change: Camera angle changed\n"
            "  - scale_change: Target scale changed significantly\n"
            "  - crowded: Many similar objects, high interference\n"
            "  - background_clutter: Complex background, hard to distinguish\n\n"

            "Step 5 - Generate Tracking Evidence (2-4 sentences in English):\n"
            "  - What is the target? (appearance, category)\n"
            "  - What is it doing? (motion, action)\n"
            "  - Why do you think this is the target? (match with memory)\n"
            "  - How does environment affect tracking?\n\n"

            "# --- OUTPUT FORMAT ---\n"
            "Respond with ONLY this JSON (no markdown fence):\n"
            "{\n"
            '  "target_status": "normal",              // Choose ONE: normal/partially_occluded/fully_occluded/out_of_view/disappeared/reappeared\n'
            '  "environment_status": ["normal"],       // Choose ALL applicable from list above\n'
            '  "bbox": [x1, y1, x2, y2],              // 0-1000 scale. [0,0,0,0] if not visible\n'
            '  "tracking_evidence": "The target is ...",  // 2-4 sentences in English\n'
            '  "confidence": 0.9                       // 0.0=lost, 1.0=certain\n'
            "}\n\n"

            "CRITICAL RULES:\n"
            "1. If target_status is fully_occluded/out_of_view/disappeared → bbox MUST be [0,0,0,0]\n"
            "2. If target_status is normal/partially_occluded/reappeared → bbox MUST be valid coordinates\n"
            "3. Search ENTIRE image, ignore BLUE box position constraint\n"
            "4. tracking_evidence MUST explain your reasoning process\n"
        )


# ==================== 认知跟踪Prompt（Mosaic 版本）====================

class CognitiveMosaicPrompt(PromptTemplate):
    """
    认知跟踪（Mosaic 版本）: 历史帧拼接 + 当前帧
    """
    def __init__(self):
        super().__init__(
            name="cognitive_mosaic_tracking",
            description="Tracking with historical frame mosaic"
        )

    def build(self,
              memory_cognition: str = "",
              language_description: str = "",
              num_history_frames: int = 2) -> str:
        return (
            "# === TASK: Cognitive Visual Tracking ===\n\n"
            "You are performing cognitive visual tracking - maintaining continuous awareness of a target object across video frames.\n\n"

            "## Input Description\n\n"
            "**Image 1 (Historical Reference Mosaic)** contains:\n"
            "- **Initial Template**: Frame #0 with GREEN bounding box (ground truth annotation)\n"
            "- **Historical Trajectory Reference**: Several historical frames with RED bounding boxes (predicted results, may contain errors)\n\n"
            "**Image 2 (Current Frame)**: Where you need to locate the target\n\n"
            f"**Current Cognitive Chain**:\n{memory_cognition}\n\n"
            f"**Initial Target Description** (optional):\n{language_description}\n\n"

            "## Your Goal\n"
            "1. Identify the target based on given information\n"
            "2. Produce one concise cognition chain that explains identity, state, and likely evolution\n"
            "3. Make the cognition chain reusable as memory for future tracking\n\n"
            "---\n\n"

            "# === OUTPUT REQUIREMENTS ===\n\n"
            "## 1. Current Frame Prediction\n\n"
            "### Target Status (choose ONE option):\n"
            "A. normal - Target clearly visible\n"
            "B. partially_occluded - Target partially blocked but identifiable\n"
            "C. fully_occluded - Target completely blocked but likely still in scene\n"
            "D. out_of_view - Target moved outside frame boundaries\n"
            "E. disappeared - Target vanished from scene\n"
            "F. reappeared - Target returned after being absent\n\n"

            "### Bounding Box:\n"
            "- If visible or location inferable (A/B/C/F): Provide [x1, y1, x2, y2] in 0-1000 scale\n"
            "- If completely unlocatable (D/E): Output [0, 0, 0, 0]\n\n"

            "### Environment Status (select ALL applicable options):\n"
            "A. normal\n"
            "B. low_light\n"
            "C. high_light\n"
            "D. motion_blur\n"
            "E. scene_change\n"
            "F. viewpoint_change\n"
            "G. scale_change\n"
            "H. crowded\n"
            "I. background_clutter\n\n"

            "## 2. Cognition Chain\n"
            "Write one concise cognition chain in 2-4 sentences:\n"
            "- First identify what the target is, grounded by Frame #0 GREEN box and the text description\n"
            "- Then describe its current state/location and why this instance matches or does not match the anchor\n"
            "- End with likely next-step evolution or tracking risk\n\n"

            "## 3. Confidence Score\n"
            "Your confidence in the prediction (0.0-1.0, 0.1 granularity)\n\n"
            "---\n\n"

            "# === OUTPUT FORMAT ===\n\n"
            "Respond with ONLY this JSON (no markdown fence):\n"
            "{\n"
            '  "target_status": "A",\n'
            '  "environment_status": ["A"],\n'
            '  "bbox": [x1, y1, x2, y2],\n'
            '  "cognition_chain": "The target is a red sedan moving right..."\n'
            '  "confidence": 0.9\n'
            "}\n"
        )


class CognitiveMosaicRefPrompt(PromptTemplate):
    """
    认知跟踪（Mosaic + Ref 版本）: 在文本中显式提供初始 GT 坐标锚点
    """
    def __init__(self):
        super().__init__(
            name="cognitive_mosaic_ref_tracking",
            description="Tracking with historical frame mosaic + explicit init bbox reference"
        )

    def build(self,
              memory_cognition: str = "",
              language_description: str = "",
              num_history_frames: int = 2,
              init_bbox_1000=None) -> str:
        return (
            "# === TASK: Cognitive Visual Tracking ===\n\n"
            "You are performing cognitive visual tracking - maintaining continuous awareness of a target object across video frames.\n\n"

            "## Input Description\n\n"
            "**Image 1 (Historical Reference Mosaic)** contains:\n"
            "- **Initial Template (ANCHOR)**: Frame #0 with GREEN bounding box (ground truth annotation)\n"
            "- **Historical Trajectory Reference**: Several historical frames with RED bounding boxes (predicted results, may contain errors)\n\n"
            f"**Explicit Initial Anchor BBox** (Image-1 coordinates, normalized [0,999]): {init_bbox_1000}\n"
            "Treat this bbox as authoritative anchor. It defines target identity.\n\n"
            "**Image 2 (Current Frame)**: Where you need to locate the target\n\n"
            f"**Current Cognitive Chain**:\n{memory_cognition}\n\n"
            f"**Initial Target Description** (optional):\n{language_description}\n\n"

            "## Priority Rules (VERY IMPORTANT)\n"
            "1. Frame #0 GREEN box + explicit anchor bbox are the ONLY authoritative identity anchors.\n"
            "2. RED boxes are motion hints only; they may be wrong.\n"
            "3. If anchor appearance conflicts with RED trajectory, trust anchor appearance.\n"
            "4. Never switch to another similar object only because RED trajectory points there.\n\n"
            "---\n\n"

            "# === OUTPUT REQUIREMENTS ===\n\n"
            "## 1. Current Frame Prediction\n\n"
            "### Target Status (choose ONE option):\n"
            "A. normal - Target clearly visible\n"
            "B. partially_occluded - Target partially blocked but identifiable\n"
            "C. fully_occluded - Target completely blocked but likely still in scene\n"
            "D. out_of_view - Target moved outside frame boundaries\n"
            "E. disappeared - Target vanished from scene\n"
            "F. reappeared - Target returned after being absent\n\n"

            "### Bounding Box:\n"
            "- If visible or location inferable (A/B/C/F): Provide [x1, y1, x2, y2] in 0-1000 scale\n"
            "- If completely unlocatable (D/E): Output [0, 0, 0, 0]\n\n"

            "### Environment Status (select ALL applicable options):\n"
            "A. normal\n"
            "B. low_light\n"
            "C. high_light\n"
            "D. motion_blur\n"
            "E. scene_change\n"
            "F. viewpoint_change\n"
            "G. scale_change\n"
            "H. crowded\n"
            "I. background_clutter\n\n"

            "## 2. Cognition Chain\n"
            "Write one concise cognition chain in 2-4 sentences:\n"
            "- First identify what the target is, grounded by Frame #0 GREEN box, the explicit anchor bbox, and the text description\n"
            "- Then describe its current state/location and why this instance matches or does not match the anchor\n"
            "- End with likely next-step evolution or tracking risk\n"
            "- Explicitly state how the anchor bbox supports your decision\n\n"

            "## 3. Confidence Score\n"
            "Your confidence in the prediction (0.0-1.0, 0.1 granularity)\n\n"

            "# === OUTPUT FORMAT ===\n\n"
            "{\n"
            '  "target_status": "A",\n'
            '  "environment_status": ["A"],\n'
            '  "bbox": [x1, y1, x2, y2],\n'
            '  "cognition_chain": "The target is a red sedan moving right..."\n'
            '  "confidence": 0.9\n'
            "}\n\n"
            "HARD CONSTRAINTS:\n"
            "- If prediction does not match Frame #0 anchor identity, output D or E with bbox [0,0,0,0] instead of forcing a wrong match.\n"
            "- Do not use RED-box consistency alone as identity evidence.\n"
        )


# ==================== 初始记忆生成Prompt（改进版）====================

class InitialMemoryPrompt(PromptTemplate):
    """
    初始记忆生成: 分析初始帧，生成discriminative语义记忆
    改进: 明确引导生成有利于长时跟踪的判别性特征描述
    """
    def __init__(self):
        super().__init__(
            name="initial_memory",
            description="Generate initial semantic memory for tracking"
        )

    def build(self, target_description: str = "") -> str:
        return (
            "# --- TASK ---\n"
            "Analyze the target object (GREEN box) and generate rich semantic memory "
            "to support long-term visual tracking.\n\n"
            f"# --- TEXT PRIOR (OPTIONAL, from dataset) ---\n"
            f"Target description hint: {target_description}\n"
            "Use it as a soft prior for category/identity, but verify with visual evidence.\n\n"

            "# --- ANALYSIS GUIDE ---\n"
            "Focus on DISCRIMINATIVE features that help re-identify this target later:\n\n"
            "1. Appearance (most important):\n"
            "   - First state target CATEGORY/IDENTITY explicitly (e.g., airplane, person, red sedan)\n"
            "   - Primary and secondary colors\n"
            "   - Shape and aspect ratio (tall/wide/square)\n"
            "   - Texture and surface pattern (solid, striped, spotted, metallic, etc.)\n"
            "   - Unique markings or distinctive details\n"
            "   - Approximate size relative to scene\n\n"
            "2. Motion State:\n"
            "   - Is the target currently moving or stationary?\n"
            "   - If moving: direction and approximate speed\n\n"
            "3. Context:\n"
            "   - Scene type (indoor/outdoor, road/field, etc.)\n"
            "   - Notable nearby objects or landmarks\n"
            "   - Target's position in the frame\n\n"

            "# --- OUTPUT FORMAT ---\n"
            "Respond with ONLY this JSON (no markdown fence):\n"
            "{\n"
            '  "appearance": "Start with category/identity, then detailed appearance in English",\n'
            '  "motion": "Current motion state in English (moving/stationary, direction, speed)",\n'
            '  "context": "Scene context in English (environment type, nearby objects, position)"\n'
            "}\n\n"

            "Example:\n"
            "{\n"
            '  "appearance": "A red sedan with white side stripes, rectangular headlights, a roof rack, and a medium-sized body.",\n'
            '  "motion": "Moving to the right at a moderate speed in a mostly straight path.",\n'
            '  "context": "An urban road scene with nearby vehicles and buildings, with the target slightly right of center."\n'
            "}\n"
        )


class InitialCognitionMosaicPrompt(PromptTemplate):
    """
    Mosaic 专用初始化认知：输出简洁、可递进更新的初始认知结果。
    """
    def __init__(self):
        super().__init__(
            name="initial_cognition_mosaic",
            description="Generate concise initial target cognition for mosaic tracking"
        )

    def build(self, target_description: str = "") -> str:
        return (
            "# === TASK: Initialize Cognitive Tracking Chain ===\n\n"
            "You are starting a cognitive visual tracking task. Your goal is to generate an initial **cognition chain** "
            "that will serve as the foundation for tracking this target across future frames.\n\n"

            "This cognition chain is NOT a generic image caption. It is a **tracking-oriented narrative** that captures:\n"
            "- What the target is (identity)\n"
            "- How to recognize it (discriminative features)\n"
            "- Where it is and what it's doing (current state)\n"
            "- What might happen next (trajectory prediction)\n\n"

            "---\n\n"

            "# === INPUT ===\n\n"
            "**Image**: One frame with a GREEN bounding box.\n"
            "- The GREEN box is an **auxiliary annotation** marking the target region for your reference.\n"
            "- Do NOT describe the green box itself (e.g., \"there is a green box\").\n"
            "- Focus on the **object inside** the green box.\n\n"

            f"**Target Text Description** (MUST consider, not optional):\n"
            f"{target_description if target_description else '(No text description provided)'}\n"
            "- This description is **mandatory input** for understanding target identity.\n"
            "- When the target is small, blurry, or ambiguous, rely heavily on this text to infer what it is.\n"
            "- Combine text description with visual evidence inside the GREEN box to form your understanding.\n\n"

            "---\n\n"

            "# === COGNITION CHAIN REQUIREMENTS ===\n\n"

            "Write a **2-4 sentence narrative** that includes:\n\n"

            "1. **Target Identity** (Sentence 1):\n"
            "   - Clearly state what the target is (e.g., \"The target is a white airplane...\")\n"
            "   - Use the text description as the primary source of identity, verified by visual evidence\n"
            "   - Avoid vague terms like \"small object\" when a specific category can be inferred\n\n"

            "2. **Discriminative Features & Spatial Context** (Sentence 2):\n"
            "   - Describe key visual features that distinguish this target from similar objects\n"
            "   - Include colors, shapes, textures, unique markings\n"
            "   - Mention spatial relationships (e.g., \"near the runway\", \"surrounded by similar vehicles\")\n"
            "   - Highlight potential distractors or confusers in the scene\n\n"

            "3. **Current State & Motion** (Sentence 3):\n"
            "   - What is the target doing right now? (moving, stationary, turning, etc.)\n"
            "   - Current position in the frame\n\n"

            "4. **Trajectory Prediction & Tracking Risks** (Sentence 4, optional):\n"
            "   - Where is the target likely to go next?\n"
            "   - What are the main tracking challenges? (occlusion risks, similar objects, lighting changes, etc.)\n\n"

            "---\n\n"

            "# === EXAMPLE ===\n\n"
            "For a tracking task with text description \"airplane\" and a white plane on a runway:\n\n"
            "\"The target is a white commercial airplane preparing to land on the runway. "
            "It has a distinctive blue tail stripe and is positioned at the far end of the runway, "
            "with another similar white airplane parked nearby on the tarmac. "
            "The target is currently descending and will likely touch down soon, then taxi toward the terminal. "
            "The main tracking risk is confusing it with the stationary plane during the landing sequence.\"\n\n"

            "---\n\n"

            "# === OUTPUT FORMAT ===\n\n"
            "Respond with ONLY this JSON (no markdown fence, no extra text):\n"
            "{\n"
            '  "init_cognition": "Your 2-4 sentence cognition chain here."\n'
            "}\n"
        )


# ==================== 认知跟踪Prompt v2（启发式推理版）====================

class CognitiveMosaicPromptV2(PromptTemplate):
    """
    认知跟踪 Mosaic v2：强调启发式推理，要求模型主动解释身份匹配证据。

    核心改进（对比 v1）：
    - 认知链不再是"按步骤描述"，而是要求"自由推理"
    - 要求模型主动分析当前帧与历史的差异（视角变化、缩放、遮挡等）
    - 通过反例引导（BAD examples）防止模型写套路化描述
    - 强调"为什么认为这还是同一个目标"
    """
    def __init__(self):
        super().__init__(
            name="cognitive_mosaic_tracking_v2",
            description="Tracking with historical frame mosaic + heuristic reasoning guidance"
        )

    def build(self,
              memory_cognition: str = "",
              language_description: str = "",
              num_history_frames: int = 2) -> str:
        return (
            "# === TASK: Cognitive Visual Tracking ===\n\n"
            "You are performing cognitive visual tracking - maintaining continuous awareness of a target object across video frames.\n\n"

            "## Input Description\n\n"
            "**Image 1 (Historical Reference Mosaic)** contains:\n"
            "- **Initial Template**: Frame #0 with GREEN bounding box (ground truth annotation)\n"
            "- **Historical Trajectory Reference**: Several historical frames with RED bounding boxes (predicted results, may contain errors)\n\n"
            "**Image 2 (Current Frame)**: Where you need to locate the target\n\n"
            f"**Previous Cognition Chain** (your last reasoning, use it as memory):\n{memory_cognition}\n\n"
            f"**Initial Target Description** (mandatory reference):\n{language_description}\n\n"
            "---\n\n"

            "# === OUTPUT REQUIREMENTS ===\n\n"
            "## 1. Current Frame Prediction\n\n"
            "### Target Status (choose ONE option):\n"
            "A. normal - Target clearly visible\n"
            "B. partially_occluded - Target partially blocked but identifiable\n"
            "C. fully_occluded - Target completely blocked but likely still in scene\n"
            "D. out_of_view - Target moved outside frame boundaries\n"
            "E. disappeared - Target vanished from scene\n"
            "F. reappeared - Target returned after being absent\n\n"

            "### Bounding Box:\n"
            "- If visible or location inferable (A/B/C/F): Provide [x1, y1, x2, y2] in 0-1000 scale\n"
            "- If completely unlocatable (D/E): Output [0, 0, 0, 0]\n\n"

            "### Environment Status (select ALL applicable options):\n"
            "A. normal\n"
            "B. low_light\n"
            "C. high_light\n"
            "D. motion_blur\n"
            "E. scene_change\n"
            "F. viewpoint_change\n"
            "G. scale_change\n"
            "H. crowded\n"
            "I. background_clutter\n\n"

            "## 2. Cognition Chain (your free-form tracking reasoning)\n\n"
            "Write 2-4 sentences of **free-form reasoning** - think like a human expert tracker.\n"
            "Your chain serves two purposes: (1) justify your current prediction, (2) become memory for the next frame.\n\n"

            "When writing, actively reason about these questions (weave answers into natural narrative, do NOT list them):\n"
            "- Is what I see in the current frame really the same target? What visual evidence confirms or challenges this?\n"
            "- Did anything change from the previous chain? (camera zoom, viewpoint shift, occlusion, re-appearance)\n"
            "  → If the target looks different now (e.g. full-body → head close-up), explicitly explain WHY it is still the same target.\n"
            "- Where is the target now, and where is it likely heading? What risk could cause tracking failure?\n\n"

            "**GOOD example** (scene: camera zooms from full-body to head close-up of a person):\n"
            "\"The camera has cut to a close-up, now showing only the head and shoulders. "
            "The short black hair and red jacket collar are consistent with the full-body view in Frame #0, "
            "confirming this is the same person despite the drastic viewpoint change. "
            "The target appears to be turning slightly left, and may exit frame from the left edge soon.\"\n\n"

            "**BAD examples** (mechanical, no reasoning - avoid these):\n"
            "- \"The target is a person. It is visible. It is in the center of the frame.\"\n"
            "- \"Target status is normal. The target is moving right.\"\n\n"

            "## 3. Confidence Score\n"
            "Your confidence in the prediction (0.0-1.0, 0.1 granularity)\n\n"
            "---\n\n"

            "# === OUTPUT FORMAT ===\n\n"
            "Respond with ONLY this JSON (no markdown fence):\n"
            "{\n"
            '  "target_status": "A",\n'
            '  "environment_status": ["A"],\n'
            '  "bbox": [x1, y1, x2, y2],\n'
            '  "cognition_chain": "Your free-form reasoning here...",\n'
            '  "confidence": 0.9\n'
            "}\n"
        )


class CognitiveMosaicRefPromptV2(PromptTemplate):
    """
    认知跟踪 Mosaic + Ref v2：在 v2 启发式推理基础上，额外提供初始 GT 坐标锚点。
    """
    def __init__(self):
        super().__init__(
            name="cognitive_mosaic_ref_tracking_v2",
            description="Tracking with historical frame mosaic + explicit init bbox reference + heuristic reasoning"
        )

    def build(self,
              memory_cognition: str = "",
              language_description: str = "",
              num_history_frames: int = 2,
              init_bbox_1000=None) -> str:
        return (
            "# === TASK: Cognitive Visual Tracking ===\n\n"
            "You are performing cognitive visual tracking - maintaining continuous awareness of a target object across video frames.\n\n"

            "## Input Description\n\n"
            "**Image 1 (Historical Reference Mosaic)** contains:\n"
            "- **Initial Template (ANCHOR)**: Frame #0 with GREEN bounding box (ground truth annotation)\n"
            "- **Historical Trajectory Reference**: Several historical frames with RED bounding boxes (predicted results, may contain errors)\n\n"
            f"**Explicit Initial Anchor BBox** (Image-1 coordinates, normalized [0,999]): {init_bbox_1000}\n"
            "Treat this bbox as the authoritative identity anchor.\n\n"
            "**Image 2 (Current Frame)**: Where you need to locate the target\n\n"
            f"**Previous Cognition Chain** (your last reasoning, use it as memory):\n{memory_cognition}\n\n"
            f"**Initial Target Description** (mandatory reference):\n{language_description}\n\n"

            "## Priority Rules\n"
            "1. Frame #0 GREEN box + explicit anchor bbox define the target identity.\n"
            "2. RED boxes are motion hints only; they may be wrong.\n"
            "3. If anchor appearance conflicts with RED trajectory, trust the anchor.\n\n"
            "---\n\n"

            "# === OUTPUT REQUIREMENTS ===\n\n"
            "## 1. Current Frame Prediction\n\n"
            "### Target Status (choose ONE option):\n"
            "A. normal - Target clearly visible\n"
            "B. partially_occluded - Target partially blocked but identifiable\n"
            "C. fully_occluded - Target completely blocked but likely still in scene\n"
            "D. out_of_view - Target moved outside frame boundaries\n"
            "E. disappeared - Target vanished from scene\n"
            "F. reappeared - Target returned after being absent\n\n"

            "### Bounding Box:\n"
            "- If visible or location inferable (A/B/C/F): Provide [x1, y1, x2, y2] in 0-1000 scale\n"
            "- If completely unlocatable (D/E): Output [0, 0, 0, 0]\n\n"

            "### Environment Status (select ALL applicable options):\n"
            "A. normal\n"
            "B. low_light\n"
            "C. high_light\n"
            "D. motion_blur\n"
            "E. scene_change\n"
            "F. viewpoint_change\n"
            "G. scale_change\n"
            "H. crowded\n"
            "I. background_clutter\n\n"

            "## 2. Cognition Chain (your free-form tracking reasoning)\n\n"
            "Write 2-4 sentences of **free-form reasoning** - think like a human expert tracker.\n"
            "Your chain serves two purposes: (1) justify your current prediction, (2) become memory for the next frame.\n\n"

            "When writing, actively reason about these questions (weave answers into natural narrative, do NOT list them):\n"
            "- Is what I see in the current frame really the same target? What visual evidence confirms or challenges this?\n"
            "- Did anything change from the previous chain? (camera zoom, viewpoint shift, occlusion, re-appearance)\n"
            "  → If the target looks different now (e.g. full-body → head close-up), explicitly explain WHY it is still the same target.\n"
            "- Where is the target now, and where is it likely heading? What risk could cause tracking failure?\n\n"

            "**GOOD example** (scene: camera zooms from full-body to head close-up of a person):\n"
            "\"The camera has cut to a close-up, now showing only the head and shoulders. "
            "The short black hair and red jacket collar are consistent with the full-body view in Frame #0, "
            "confirming this is the same person despite the drastic viewpoint change. "
            "The target appears to be turning slightly left, and may exit frame from the left edge soon.\"\n\n"

            "**BAD examples** (mechanical, no reasoning - avoid these):\n"
            "- \"The target is a person. It is visible. It is in the center of the frame.\"\n"
            "- \"Target status is normal. The target is moving right.\"\n\n"

            "## 3. Confidence Score\n"
            "Your confidence in the prediction (0.0-1.0, 0.1 granularity)\n\n"
            "---\n\n"

            "# === OUTPUT FORMAT ===\n\n"
            "Respond with ONLY this JSON (no markdown fence):\n"
            "{\n"
            '  "target_status": "A",\n'
            '  "environment_status": ["A"],\n'
            '  "bbox": [x1, y1, x2, y2],\n'
            '  "cognition_chain": "Your free-form reasoning here...",\n'
            '  "confidence": 0.9\n'
            "}\n"
        )

class InitialCognitionMosaicPromptV2(PromptTemplate):
    """
    Mosaic 初始化认知 V2：与 V1 结构相同，但强调推理预判而非特征罗列。
    引导模型像人类一样"先想清楚再跟踪"。
    """
    def __init__(self):
        super().__init__(
            name="initial_cognition_mosaic_v2",
            description="Initial cognition chain with reasoning emphasis (V2)"
        )

    def build(self, target_description: str = "") -> str:
        return (
            "# === TASK: Initialize Cognitive Tracking Chain ===\n\n"
            "You are starting a cognitive visual tracking task. Generate an initial **cognition chain** — "
            "a tracking-oriented narrative that will be passed to future frames as your memory.\n\n"

            "---\n\n"

            "# === INPUT ===\n\n"
            "**Image**: One frame with a GREEN bounding box.\n"
            "- The GREEN box is an **auxiliary annotation** marking the target region.\n"
            "- Do NOT describe the green box itself. Focus on the **object inside**.\n\n"
            f"**Target Text Description** (MUST consider, not optional):\n"
            f"{target_description if target_description else '(No text description provided)'}\n"
            "- This is **mandatory input** for understanding target identity.\n"
            "- When the target is small, blurry, or ambiguous, rely heavily on this text.\n\n"

            "---\n\n"

            "# === WHAT TO WRITE ===\n\n"
            "Write a **2-4 sentence narrative** as if you are a human tracker preparing to follow this target. "
            "Think about:\n\n"
            "- **Who/what am I tracking?** State clearly, combining text description + visual evidence.\n"
            "- **How will I recognize it later?** What makes it unique? What could I confuse it with?\n"
            "- **What is it doing right now?** Position, motion, action.\n"
            "- **What should I watch out for?** Predict what might happen and what could go wrong.\n\n"

            "Example — tracking an airplane (text description: \"airplane\"):\n\n"
            "\"The target is a white commercial airplane with a blue tail stripe, currently descending toward "
            "the runway at the far end. A similar white airplane is parked on the tarmac to the right — "
            "I need to track the one in motion, not the stationary one. "
            "The target will likely touch down soon and begin taxiing, at which point the two planes "
            "could be closer together and harder to distinguish.\"\n\n"

            "Example — tracking a person (text description: \"person\"):\n\n"
            "\"The target is a woman in a red jacket and jeans, standing near the entrance of a café. "
            "Several other pedestrians are nearby, but she is the only one wearing red. "
            "She appears to be about to walk inside, which would cause a full occlusion. "
            "If the camera follows, I should expect a scene change from outdoor to indoor.\"\n\n"

            "---\n\n"

            "# === OUTPUT FORMAT ===\n\n"
            "Respond with ONLY this JSON (no markdown fence, no extra text):\n"
            "{\n"
            '  "init_cognition": "Your 2-4 sentence cognition chain here."\n'
            "}\n"
        )


class PromptManager:
    """
    Prompt管理器：统一管理所有prompt模板
    """

    _prompts = {
        # 跟踪prompt
        "two_image": TwoImageTrackingPrompt(),
        "three_image": ThreeImageTrackingPrompt(),
        "cognitive": CognitiveTrackingPrompt(),  # 认知跟踪（统一框架）
        "cognitive_mosaic": CognitiveMosaicPrompt(),  # 认知跟踪（Mosaic 版本）
        "cognitive_mosaic_ref": CognitiveMosaicRefPrompt(),  # 认知跟踪（Mosaic + 坐标锚点）
        "cognitive_mosaic_v2": CognitiveMosaicPromptV2(),  # v2：自由启发式推理
        "cognitive_mosaic_ref_v2": CognitiveMosaicRefPromptV2(),  # v2：自由推理 + 坐标锚点

        # 辅助prompt
        "init_memory": InitialMemoryPrompt(),
        "init_cognition_mosaic": InitialCognitionMosaicPrompt(),       # v1
        "init_cognition_mosaic_v2": InitialCognitionMosaicPromptV2(),  # v2：推理预判
        "init_story_mosaic": InitialCognitionMosaicPrompt(),           # 兼容旧名称
    }

    @classmethod
    def get(cls, prompt_name: str) -> PromptTemplate:
        """
        获取prompt模板

        Args:
            prompt_name: prompt名称，可选:
                - "two_image": 两图跟踪
                - "three_image": 三图跟踪
                - "cognitive": 认知跟踪
                - "cognitive_mosaic": 认知跟踪（Mosaic 版本）v1
                - "cognitive_mosaic_ref": 认知跟踪（Mosaic + 坐标锚点）v1
                - "cognitive_mosaic_v2": 认知跟踪（Mosaic 版本）v2，自由启发式推理
                - "cognitive_mosaic_ref_v2": 认知跟踪（Mosaic + 坐标锚点）v2，自由启发式推理
                - "init_memory": 初始记忆生成
                - "init_cognition_mosaic": Mosaic专用初始认知结果 v1
                - "init_cognition_mosaic_v2": Mosaic专用初始认知结果 v2，推理预判
                - "init_story_mosaic": 兼容旧名称，等价于 init_cognition_mosaic

        Returns:
            PromptTemplate实例
        """
        if prompt_name not in cls._prompts:
            available = list(cls._prompts.keys())
            raise ValueError(
                f"Unknown prompt name: {prompt_name}. "
                f"Available prompts: {available}"
            )
        return cls._prompts[prompt_name]

    @classmethod
    def list_prompts(cls):
        """列出所有可用的prompt"""
        print("Available Prompts:")
        for name, prompt in cls._prompts.items():
            print(f"  - {name}: {prompt.description}")

    @classmethod
    def register(cls, name: str, prompt: PromptTemplate):
        """注册新的prompt模板"""
        cls._prompts[name] = prompt


# ==================== 便捷函数 ====================

def get_prompt(prompt_name: str, **kwargs) -> str:
    """
    便捷函数：获取并构建prompt

    Args:
        prompt_name: prompt名称
        **kwargs: 传递给prompt的参数

    Returns:
        构建好的prompt字符串

    Example:
        >>> prompt = get_prompt("two_image", target_description="a red car")
        >>> prompt = get_prompt("cognitive",
        ...                     memory_appearance="red sedan",
        ...                     memory_motion="moving right")
    """
    template = PromptManager.get(prompt_name)
    return template.build(**kwargs)


# ==================== 示例使用 ====================

if __name__ == "__main__":
    # 列出所有prompt
    PromptManager.list_prompts()

    # 使用示例
    print("\n" + "="*60)
    print("Two-Image Tracking Prompt:")
    print("="*60)
    prompt = get_prompt("two_image", target_description="a red car")
    print(prompt)

    print("\n" + "="*60)
    print("Three-Image Tracking Prompt:")
    print("="*60)
    prompt = get_prompt("three_image", target_description="a person in blue shirt")
    print(prompt)

    print("\n" + "="*60)
    print("Cognitive Tracking Prompt:")
    print("="*60)
    prompt = get_prompt("cognitive",
                       memory_appearance="red sedan with chrome trim",
                       memory_motion="moving rightward at moderate speed",
                       memory_context="highway with other vehicles")
    print(prompt)

    print("\n" + "="*60)
    print("Init Memory Prompt:")
    print("="*60)
    prompt = get_prompt("init_memory")
    print(prompt)

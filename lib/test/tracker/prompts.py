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


# ==================== 记忆库跟踪Prompt ====================

class MemoryBankPrompt(PromptTemplate):
    """
    记忆库跟踪: 语义记忆 + 上一帧(蓝框) + 当前帧
    改进: 引导关注外观变化(光照/尺度/形变)、结构化记忆更新
    """
    def __init__(self):
        super().__init__(
            name="memory_bank_tracking",
            description="Tracking with semantic memory bank"
        )

    def build(self,
              memory_appearance: str = "",
              memory_motion: str = "",
              memory_context: str = "") -> str:
        return (
            "# --- TASK ---\n"
            "Track a target using semantic memory and recent motion. The target may have changed "
            "appearance due to lighting, scale, or deformation.\n\n"

            "# --- SEMANTIC MEMORY (accumulated from previous frames) ---\n"
            f"Appearance: {memory_appearance}\n"
            f"Motion: {memory_motion}\n"
            f"Context: {memory_context}\n\n"

            "# --- INPUTS ---\n"
            "Image 1 (Previous Frame - BLUE box): Recent position. Use for motion estimation.\n"
            "Image 2 (Current Frame): Locate the target here.\n\n"

            "# --- TRACKING STRATEGY ---\n"
            "Step 1 - Recall target from memory:\n"
            "  - Color/texture may shift due to lighting changes — allow some tolerance\n"
            "  - Size may change due to camera zoom or depth change\n"
            "  - Shape may deform (e.g., person walking, animal moving)\n\n"
            "Step 2 - Estimate motion from Image 1 (BLUE box position):\n"
            "  - Predict where target has moved in Image 2\n\n"
            "Step 3 - Find best match in Image 2:\n"
            "  - Combine memory description + motion prediction\n"
            "  - If target is not present: output [0,0,0,0]\n\n"
            "Step 4 - Update semantic memory with current observation.\n\n"

            "# --- OUTPUT FORMAT ---\n"
            "Respond with ONLY this JSON:\n"
            "{\n"
            '  "bbox": [x1, y1, x2, y2],  // 0-1000 scale. [0,0,0,0] if not visible.\n'
            '  "evidence": "How memory matched and motion aligned (1-2 sentences)",\n'
            '  "confidence": 0.9,\n'
            '  "state": {\n'
            '    "appearance": "Updated appearance (note any changes from memory)",\n'
            '    "motion": "Current motion direction and estimated speed",\n'
            '    "context": "Current surrounding scene and relative position"\n'
            '  }\n'
            "}\n"
        )


# ==================== 初始记忆生成Prompt ====================

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

    def build(self) -> str:
        return (
            "# --- TASK ---\n"
            "Analyze the target object (GREEN box) and generate rich semantic memory "
            "to support long-term visual tracking.\n\n"

            "# --- ANALYSIS GUIDE ---\n"
            "Focus on DISCRIMINATIVE features that help re-identify this target later:\n\n"
            "1. Appearance (most important):\n"
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
            "Respond with ONLY this JSON:\n"
            "{\n"
            '  "appearance": "Detailed description emphasizing discriminative features for re-identification",\n'
            '  "motion": "Current motion state (stationary / moving direction + speed)",\n'
            '  "context": "Scene type and spatial context"\n'
            "}\n"
        )


# ==================== Prompt管理器 ====================

class PromptManager:
    """
    Prompt管理器：统一管理所有prompt模板
    """

    _prompts = {
        # 跟踪prompt
        "two_image": TwoImageTrackingPrompt(),
        "three_image": ThreeImageTrackingPrompt(),
        "memory_bank": MemoryBankPrompt(),

        # 辅助prompt
        "init_memory": InitialMemoryPrompt(),
    }

    @classmethod
    def get(cls, prompt_name: str) -> PromptTemplate:
        """
        获取prompt模板

        Args:
            prompt_name: prompt名称，可选:
                - "two_image": 两图跟踪
                - "three_image": 三图跟踪
                - "memory_bank": 记忆库跟踪
                - "init_memory": 初始记忆生成

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
        >>> prompt = get_prompt("memory_bank",
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
    print("Memory Bank Tracking Prompt:")
    print("="*60)
    prompt = get_prompt("memory_bank",
                       memory_appearance="red sedan with chrome trim",
                       memory_motion="moving rightward at moderate speed",
                       memory_context="highway with other vehicles")
    print(prompt)

    print("\n" + "="*60)
    print("Init Memory Prompt:")
    print("="*60)
    prompt = get_prompt("init_memory")
    print(prompt)

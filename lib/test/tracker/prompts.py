"""
VLM Tracker Prompt配置管理模块

统一管理所有tracker的prompt，支持通过配置名称读取不同的prompt模板。
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
    两图跟踪: 模板帧 + 当前帧
    """
    def __init__(self):
        super().__init__(
            name="two_image_tracking",
            description="Template image with bbox + current image"
        )
    
    def build(self, target_description: str = "the target object") -> str:
        return (
            "# --- CORE TASK ---\n"
            "Track the target object across frames. Determine if the target is still visible and locate it.\n\n"
            
            "# --- VISUAL REFERENCE ---\n"
            f"Image 1 (Template): Target marked by GREEN box. Target is: {target_description}.\n"
            "Image 2 (Current): Find the same target here.\n\n"
            
            "# --- OUTPUT REQUIREMENT ---\n"
            "Output JSON format:\n"
            "{\n"
            '  "bbox": [x1, y1, x2, y2],      // 0-1000 scale. Output [0,0,0,0] if target is invisible/occluded.\n'
            '  "evidence": "Briefly describe the visual features matched (e.g., \'Matched red color and shape\') and spatial logic (e.g., \'Moved slightly right\').",\n'
            '  "confidence": 0.95             // A float between 0.0 (Lost) and 1.0 (Certain).\n'
            "}\n"
        )


# ==================== 三图跟踪Prompt ====================

class ThreeImageTrackingPrompt(PromptTemplate):
    """
    三图跟踪: 初始帧 + 上一帧 + 当前帧
    """
    def __init__(self):
        super().__init__(
            name="three_image_tracking",
            description="Initial frame + previous frame + current frame"
        )
    
    def build(self, target_description: str = "the target object") -> str:
        return (
            "# --- CORE TASK ---\n"
            "Track the target using initial appearance and motion cues. Determine if target is visible and locate it.\n\n"
            
            "# --- VISUAL REFERENCE ---\n"
            f"Image 1 (Initial - GREEN box): Ground truth target. Target is: {target_description}.\n"
            "Image 2 (Previous - BLUE box): Last prediction (may be inaccurate, use only for motion reference).\n"
            "Image 3 (Current): Find the target here.\n\n"
            
            "# --- OUTPUT REQUIREMENT ---\n"
            "Match the target based on: (1) Initial appearance (Image 1), (2) Motion trend (Image 2).\n"
            "Output JSON format:\n"
            "{\n"
            '  "bbox": [x1, y1, x2, y2],      // 0-1000 scale. Output [0,0,0,0] if target is invisible/occluded.\n'
            '  "evidence": "Describe matched features from Image 1 and motion from Image 2.",\n'
            '  "confidence": 0.95             // Float between 0.0 (Lost) and 1.0 (Certain).\n'
            "}\n"
        )


# ==================== 记忆库跟踪Prompt ====================

class MemoryBankPrompt(PromptTemplate):
    """
    记忆库跟踪: 使用语义记忆辅助跟踪
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
            "# --- CORE TASK ---\n"
            "Track the target using semantic memory and motion cues. Determine if target is visible and locate it.\n\n"
            
            "# --- SEMANTIC MEMORY ---\n"
            f"Appearance: {memory_appearance}\n"
            f"Motion: {memory_motion}\n"
            f"Context: {memory_context}\n\n"
            
            "# --- VISUAL REFERENCE ---\n"
            "Image 1 (Previous - BLUE box): Last prediction (may be inaccurate, use only for motion reference).\n"
            "Image 2 (Current): Find the target here.\n\n"
            
            "# --- OUTPUT REQUIREMENT ---\n"
            "Match the target based on: (1) Semantic memory, (2) Motion from Image 1.\n"
            "Output JSON format with TWO fields:\n"
            "{\n"
            '  "bbox": [x1, y1, x2, y2],      // 0-1000 scale. Output [0,0,0,0] if target is invisible/occluded.\n'
            '  "evidence": "Describe matched features from memory and observed motion.",\n'
            '  "confidence": 0.95,            // Float between 0.0 (Lost) and 1.0 (Certain).\n'
            '  "state": {                     // Update current state for memory.\n'
            '    "appearance": "current appearance description",\n'
            '    "motion": "current motion state",\n'
            '    "context": "current context"\n'
            '  }\n'
            "}\n"
        )


class InitialMemoryPrompt(PromptTemplate):
    """
    初始记忆生成Prompt
    """
    def __init__(self):
        super().__init__(
            name="initial_memory",
            description="Generate initial semantic memory"
        )
    
    def build(self) -> str:
        return (
            "# --- TASK ---\n"
            "Analyze the target object marked by the GREEN box.\n\n"
            
            "# --- OUTPUT ---\n"
            "Provide a detailed description in JSON:\n"
            "{\n"
            '  "appearance": "color, shape, texture, distinctive features",\n'
            '  "motion": "current motion state",\n'
            '  "context": "surrounding objects and position"\n'
            "}\n"
            "Be specific. Output ONLY the JSON object.\n"
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

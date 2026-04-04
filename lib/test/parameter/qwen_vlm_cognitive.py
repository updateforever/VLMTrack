"""
Parameters for QwenVLMCognitive - 认知跟踪器

yaml_name 示例:
    default     → API模式
    local_4b    → 本地4B模型

使用方法:
    python tracking/test.py qwen_vlm_cognitive default   --dataset_name lasot
    python tracking/test.py qwen_vlm_cognitive local_4b  --dataset_name lasot
"""
from lib.test.utils import TrackerParams
from lib.test.evaluation.environment import env_settings
from lib.test.parameter.vlm_common import apply_vlm_config


def parameters(yaml_name: str = "default"):
    params = TrackerParams()
    env = env_settings()

    apply_vlm_config(params, yaml_name)

    # Prompt 配置（使用新的认知跟踪 prompt）
    params.track_prompt = 'cognitive'  # 新版结构化认知跟踪
    params.init_prompt = 'init_memory'

    # 调试与日志
    params.debug = 0

    # 稀疏关键帧推理
    params.use_keyframe = True
    params.keyframe_root = getattr(env, 'keyframe_root', '')

    # 框架兼容
    params.checkpoint = None
    params.save_all_boxes = False

    return params

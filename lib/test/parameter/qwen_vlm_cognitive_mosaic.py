"""
Parameters for QwenVLMCognitiveMosaic - 认知跟踪器（Mosaic 版本）

yaml_name 示例:
    default     → API模式, buffer_size=2, sample_interval=30
    local_4b    → 本地4B模型
    default_b3  → buffer_size=3
    default_s20 → sample_interval=20

使用方法:
    python tracking/test.py qwen_vlm_cognitive_mosaic default   --dataset lasot
    python tracking/test.py qwen_vlm_cognitive_mosaic local_4b  --dataset lasot
    python tracking/test.py qwen_vlm_cognitive_mosaic default_b3_s20  --dataset lasot
"""
from lib.test.utils import TrackerParams
from lib.test.evaluation.environment import env_settings
from lib.test.parameter.vlm_common import apply_vlm_config


def parameters(yaml_name: str = "default"):
    params = TrackerParams()
    env = env_settings()

    # 解析 yaml_name 后缀（如 default_b3_s20）
    tokens = yaml_name.split('_')
    base_name = tokens[0]
    buffer_size = 3
    sample_interval = 30

    for token in tokens[1:]:
        if token.startswith('b') and token[1:].isdigit():
            buffer_size = int(token[1:])
        elif token.startswith('s') and token[1:].isdigit():
            sample_interval = int(token[1:])

    # 应用 VLM 配置
    apply_vlm_config(params, base_name)

    # Mosaic 专用配置
    params.history_buffer_size = buffer_size
    params.sample_interval = sample_interval

    # Prompt 配置
    params.track_prompt = 'cognitive_mosaic'  # Mosaic 专用 prompt
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

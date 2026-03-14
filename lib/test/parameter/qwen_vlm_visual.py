"""
Parameters for QwenVLMVisual - 纯视觉跟踪器

yaml_name 后缀约定:
    {model}[_{N}f]

示例:
    default / default_2f  → API模式，两图
    default_3f            → API模式，三图
    local_4b_3f           → 本地4B，三图
    local_8b_2f           → 本地8B，两图

使用方法:
    python tracking/test.py qwen_vlm_visual default      --dataset_name lasot
    python tracking/test.py qwen_vlm_visual local_4b_3f  --dataset_name lasot
"""
from lib.test.utils import TrackerParams
from lib.test.evaluation.environment import env_settings
from lib.test.parameter.vlm_common import apply_vlm_config


def parameters(yaml_name: str = "default"):
    params = TrackerParams()
    env = env_settings()

    # 解析 yaml_name 后缀，应用 VLM 模型配置
    extras = apply_vlm_config(params, yaml_name)

    # num_frames: 后缀指定 > 参数默认值 2
    params.num_frames = extras.get('num_frames', 2)
    params.prompt_name = 'two_image' if params.num_frames == 2 else 'three_image'

    # 调试与日志
    params.debug = 0  # 0=静默, 1=控制台, 2=保存可视化

    # 稀疏关键帧推理（由 tracker.py 控制，tracker 内部 last-frame fallback）
    params.use_keyframe = True
    params.keyframe_root = getattr(env, 'keyframe_root', '')

    # 框架兼容
    params.checkpoint = None
    params.save_all_boxes = False

    return params

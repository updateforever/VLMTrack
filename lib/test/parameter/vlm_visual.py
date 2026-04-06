"""
Parameters for vlm_visual

职责约定:
    - tracker_name: 跟踪范式（vlm_visual）
    - tracker_param: 部署方式+模型（api_xxx / local_xxx）
"""
from lib.test.utils import TrackerParams
from lib.test.evaluation.environment import env_settings
from lib.test.parameter.vlm_common import (
    apply_vlm_config,
    load_vlm_experiment_defaults,
    apply_param_dict,
)


def parameters(tracker_param: str = "api_default"):
    params = TrackerParams()
    env = env_settings()

    # 1) 先加载范式默认参数（yaml）
    defaults = load_vlm_experiment_defaults('vlm_visual')
    apply_param_dict(params, defaults)

    # 2) 再根据 tracker_param 覆盖部署方式/模型
    apply_vlm_config(params, tracker_param)

    # 3) 保底兼容字段
    params.num_frames = getattr(params, 'num_frames', 2)
    params.prompt_name = getattr(params, 'prompt_name', 'two_image' if params.num_frames == 2 else 'three_image')
    params.debug = getattr(params, 'debug', 0)
    params.use_keyframe = getattr(params, 'use_keyframe', True)
    params.keyframe_root = getattr(params, 'keyframe_root', getattr(env, 'keyframe_root', ''))
    params.checkpoint = getattr(params, 'checkpoint', None)
    params.save_all_boxes = getattr(params, 'save_all_boxes', False)

    return params

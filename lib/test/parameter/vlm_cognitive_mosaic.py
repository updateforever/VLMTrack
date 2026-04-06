"""
Parameters for vlm_cognitive_mosaic

tracker_param 语法:
    <deploy_and_model>[_bN][_sM][_ref]

deploy_and_model:
    - api_default / api_xxx
    - local_qwen35_9b / local_qwen3vl_4b_thinking / local_4b ...
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
    defaults = load_vlm_experiment_defaults('vlm_cognitive_mosaic')
    apply_param_dict(params, defaults)

    # 2) 解析 mosaic 专属后缀（bN/sM），其余作为 deploy_and_model
    tokens = tracker_param.lower().split('_')
    core_tokens = list(tokens)

    buffer_size = getattr(params, 'history_buffer_size', 3)
    sample_interval = getattr(params, 'sample_interval', 30)
    use_init_bbox_ref = getattr(params, 'use_init_bbox_ref', False)

    while core_tokens:
        token = core_tokens[-1]
        if token.startswith('b') and token[1:].isdigit():
            buffer_size = int(token[1:])
            core_tokens.pop()
            continue
        if token.startswith('s') and token[1:].isdigit():
            sample_interval = int(token[1:])
            core_tokens.pop()
            continue
        if token in ('ref', 'coord', 'anchor'):
            use_init_bbox_ref = True
            core_tokens.pop()
            continue
        break

    deploy_and_model = '_'.join(core_tokens) if core_tokens else 'api_default'

    # 3) 覆盖部署方式/模型
    apply_vlm_config(params, deploy_and_model)

    # 4) 覆盖 mosaic 参数
    params.history_buffer_size = buffer_size
    params.sample_interval = sample_interval
    params.use_init_bbox_ref = use_init_bbox_ref

    # 5) 保底兼容字段
    default_prompt = 'cognitive_mosaic_ref' if use_init_bbox_ref else 'cognitive_mosaic'
    params.track_prompt = getattr(params, 'track_prompt', default_prompt)
    if use_init_bbox_ref:
        params.track_prompt = 'cognitive_mosaic_ref'
    params.init_prompt = getattr(params, 'init_prompt', 'init_story_mosaic')
    params.debug = getattr(params, 'debug', 0)
    params.use_keyframe = getattr(params, 'use_keyframe', True)
    params.keyframe_root = getattr(params, 'keyframe_root', getattr(env, 'keyframe_root', ''))
    params.checkpoint = getattr(params, 'checkpoint', None)
    params.save_all_boxes = getattr(params, 'save_all_boxes', False)

    return params

"""
VLM Tracker 公共参数模块

所有基于 QwenVLM 的 tracker 参数文件共用此模块，避免重复定义。

MODEL_CONFIGS 格式:
    key → {'mode': 'local'|'api', ...}

yaml_name 后缀约定 (qwen_vlm_visual / qwen_vlm_hybrid):
    {model_key}[_{N}f]        -- N帧, e.g. default_2f / local_4b_3f
    {model_key}_{vlm}_{trig}  -- hybrid专用, e.g. default_visual_kf
"""
import os

# ========== 模型配置字典 ==========

MODEL_CONFIGS = {
    # 本地模型
    'local_4b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen3-VL-4B-Instruct',
    },
    'local_8b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen3-VL-8B-Instruct',
    },
    # API 模型
    'api': {
        'mode': 'api',
        'api_model': 'qwen3-vl-235b-a22b-instruct',
        'api_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    },
    # 默认 (alias for 'api')
    'default': {
        'mode': 'api',
        'api_model': 'qwen3-vl-235b-a22b-instruct',
        'api_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    },
}


# ========== 后缀解析工具 ==========

def _parse_model_key(token: str) -> str:
    """将 token 匹配到 MODEL_CONFIGS 的 key，支持模糊匹配。"""
    if token in MODEL_CONFIGS:
        return token
    # 支持 'local4b' 等常见变体
    for key in MODEL_CONFIGS:
        if key.replace('_', '') == token.replace('_', ''):
            return key
    return 'default'


def parse_yaml_name(yaml_name: str):
    """
    解析 yaml_name，返回 (model_key, extras_dict)。

    qwen_vlm_visual 示例:
        'default'       → ('default', {})
        'default_2f'    → ('default', {'num_frames': 2})
        'local_4b_3f'   → ('local_4b', {'num_frames': 3})

    qwen_vlm_hybrid 示例:
        'default_visual_kf'       → ('default', {'vlm_mode': 'visual',    'trigger_mode': 'keyframe'})
        'default_cognitive_conf'  → ('default', {'vlm_mode': 'cognitive', 'trigger_mode': 'confidence'})
        'local_4b_visual_hybrid'  → ('local_4b', {'vlm_mode': 'visual',   'trigger_mode': 'hybrid'})
    """
    tokens = yaml_name.lower().split('_')
    extras = {}

    # 从后往前消费已知的后缀 token
    remaining = list(tokens)

    # trigger_mode 后缀
    _trigger_map = {'kf': 'keyframe', 'keyframe': 'keyframe',
                    'conf': 'confidence', 'confidence': 'confidence',
                    'hybrid': 'hybrid'}
    if remaining and remaining[-1] in _trigger_map:
        extras['trigger_mode'] = _trigger_map[remaining.pop()]

    # vlm_mode 后缀
    _vlm_map = {'visual': 'visual', 'cognitive': 'cognitive'}
    if remaining and remaining[-1] in _vlm_map:
        extras['vlm_mode'] = _vlm_map[remaining.pop()]

    # num_frames 后缀: '2f' / '3f'
    if remaining and remaining[-1].endswith('f') and remaining[-1][:-1].isdigit():
        extras['num_frames'] = int(remaining.pop()[:-1])

    # 剩余 token 拼成 model_key
    model_key = '_'.join(remaining) if remaining else 'default'
    model_key = _parse_model_key(model_key)

    return model_key, extras


# ========== 参数应用函数 ==========

def apply_vlm_config(params, yaml_name: str):
    """
    根据 yaml_name 将 VLM 模型参数写入 params 对象。

    Args:
        params:    TrackerParams 实例
        yaml_name: 配置名称字符串（支持后缀约定）

    Returns:
        extras (dict): 未被本函数消费的额外参数（如 num_frames, vlm_mode 等）
    """
    model_key, extras = parse_yaml_name(yaml_name)
    config = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS['default'])

    params.mode = config['mode']

    if params.mode == 'local':
        params.model_name = config.get('model_name', 'Qwen/Qwen3-VL-4B-Instruct')
        params.model_path = config.get('model_path', None)
    else:
        params.api_model = config.get('api_model', 'qwen3-vl-235b-a22b-instruct')
        params.api_base_url = config.get('api_base_url',
                                         'https://dashscope.aliyuncs.com/compatible-mode/v1')
        params.api_key = os.environ.get('DASHSCOPE_API_KEY', '')

    return extras

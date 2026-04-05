"""
VLM Tracker 公共参数模块

所有基于 QwenVLM 的 tracker 参数文件共用此模块，避免重复定义。

MODEL_CONFIGS 格式:
    key → {'mode': 'local'|'api', ...}

yaml_name 后缀约定 (qwen_vlm_visual / qwen_vlm_hybrid):
    {model_key}[_{N}f]        -- N帧, e.g. default_2f / qwen25_7b_3f
    {model_key}_{vlm}_{trig}  -- hybrid专用, e.g. default_visual_kf
"""
import os

# 固定本机模型路径（跨服务器切换时手动修改这里）
_MLLM_ROOT = '/root/user-data/MODEL_WEIGHTS_PUBLIC/MLLM_weights'

# Qwen2.5-VL
_QWEN25_3B_PATH = os.path.join(_MLLM_ROOT, 'Qwen2_5-VL-3B-Instruct')
_QWEN25_7B_PATH = os.path.join(_MLLM_ROOT, 'Qwen2_5-VL-7B-Instruct')
_QWEN25_32B_PATH = os.path.join(_MLLM_ROOT, 'Qwen2_5-VL-32B-Instruct')

# Qwen3-VL
_QWEN3VL_2B_PATH = os.path.join(_MLLM_ROOT, 'Qwen3-VL-2B-Instruct')
_QWEN3VL_4B_PATH = os.path.join(_MLLM_ROOT, 'Qwen3-VL-4B-Instruct')
_QWEN3VL_4B_THINKING_PATH = os.path.join(_MLLM_ROOT, 'Qwen3-VL-4B-Thinking')
_QWEN3VL_8B_PATH = os.path.join(_MLLM_ROOT, 'Qwen3-VL-8B-Instruct')
_QWEN3VL_32B_PATH = os.path.join(_MLLM_ROOT, 'Qwen3-VL-32B-Instruct')

# Qwen3.5
_QWEN35_4B_PATH = os.path.join(_MLLM_ROOT, 'Qwen3_5-4B')
_QWEN35_9B_PATH = os.path.join(_MLLM_ROOT, 'Qwen3_5-9B')
_QWEN35_27B_PATH = os.path.join(_MLLM_ROOT, 'Qwen3_5-27B')

# ========== 模型配置字典 ==========

MODEL_CONFIGS = {
    # ------- Qwen2.5-VL -------
    'qwen25_3b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen2.5-VL-3B-Instruct',
        'model_path': _QWEN25_3B_PATH,
    },
    'qwen25_7b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen2.5-VL-7B-Instruct',
        'model_path': _QWEN25_7B_PATH,
    },
    'qwen25_32b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen2.5-VL-32B-Instruct',
        'model_path': _QWEN25_32B_PATH,
    },

    # ------- Qwen3-VL -------
    'qwen3vl_2b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen3-VL-2B-Instruct',
        'model_path': _QWEN3VL_2B_PATH,
    },
    'qwen3vl_4b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen3-VL-4B-Instruct',
        'model_path': _QWEN3VL_4B_PATH,
    },
    'qwen3vl_4b_thinking': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen3-VL-4B-Thinking',
        'model_path': _QWEN3VL_4B_THINKING_PATH,
    },
    'qwen3vl_8b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen3-VL-8B-Instruct',
        'model_path': _QWEN3VL_8B_PATH,
    },
    'qwen3vl_32b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen3-VL-32B-Instruct',
        'model_path': _QWEN3VL_32B_PATH,
    },

    # ------- Qwen3.5 -------
    # 注意: Qwen3.5 路径需要你本地已下载对应模型目录
    'qwen35_4b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen3.5-4B',
        'model_path': _QWEN35_4B_PATH,
    },
    'qwen35_9b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen3.5-9B',
        'model_path': _QWEN35_9B_PATH,
    },
    'qwen35_27b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen3.5-27B',
        'model_path': _QWEN35_27B_PATH,
    },

    # ------- 兼容旧命名 -------
    # local_3b/local_7b: 历史上常用于 Qwen2.5-VL
    'local_3b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen2.5-VL-3B-Instruct',
        'model_path': _QWEN25_3B_PATH,
    },
    'local_7b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen2.5-VL-7B-Instruct',
        'model_path': _QWEN25_7B_PATH,
    },
    # local_4b/local_8b: 兼容历史脚本
    # 注意: 你当前机器上已存在 4b-thinking，因此 local_4b 默认指向它
    'local_4b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen3-VL-4B-Thinking',
        'model_path': _QWEN3VL_4B_THINKING_PATH,
    },
    'local_8b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen3-VL-8B-Instruct',
        'model_path': _QWEN3VL_8B_PATH,
    },

    'api': {
        'mode': 'api',
        'api_model': 'qwen3-vl-235b-a22b-instruct',
        'api_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    },
    'default': {
        'mode': 'api',
        'api_model': 'qwen3-vl-235b-a22b-instruct',
        'api_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    },
}


def _parse_model_key(token: str) -> str:
    """将 token 匹配到 MODEL_CONFIGS 的 key，支持模糊匹配。"""
    if token in MODEL_CONFIGS:
        return token
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
        'qwen25_7b_3f'  → ('qwen25_7b', {'num_frames': 3})

    qwen_vlm_hybrid 示例:
        'default_visual_kf'       → ('default', {'vlm_mode': 'visual',    'trigger_mode': 'keyframe'})
        'default_cognitive_conf'  → ('default', {'vlm_mode': 'cognitive', 'trigger_mode': 'confidence'})
        'qwen3vl_4b_visual_hybrid'  → ('qwen3vl_4b', {'vlm_mode': 'visual',   'trigger_mode': 'hybrid'})
    """
    tokens = yaml_name.lower().split('_')
    extras = {}
    remaining = list(tokens)

    _trigger_map = {'kf': 'keyframe', 'keyframe': 'keyframe',
                    'conf': 'confidence', 'confidence': 'confidence',
                    'hybrid': 'hybrid'}
    if remaining and remaining[-1] in _trigger_map:
        extras['trigger_mode'] = _trigger_map[remaining.pop()]

    _vlm_map = {'visual': 'visual', 'cognitive': 'cognitive'}
    if remaining and remaining[-1] in _vlm_map:
        extras['vlm_mode'] = _vlm_map[remaining.pop()]

    if remaining and remaining[-1].endswith('f') and remaining[-1][:-1].isdigit():
        extras['num_frames'] = int(remaining.pop()[:-1])

    model_key = '_'.join(remaining) if remaining else 'default'
    model_key = _parse_model_key(model_key)
    return model_key, extras


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
        params.model_name = config.get('model_name', 'Qwen/Qwen2.5-VL-7B-Instruct')
        params.model_path = config.get('model_path', None)
    else:
        params.api_model = config.get('api_model', 'qwen3-vl-235b-a22b-instruct')
        params.api_base_url = config.get('api_base_url',
                                         'https://dashscope.aliyuncs.com/compatible-mode/v1')
        params.api_key = os.environ.get('DASHSCOPE_API_KEY', '')

    return extras

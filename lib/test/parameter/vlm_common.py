"""
VLM Tracker 公共参数模块

所有基于 QwenVLM 的 tracker 参数文件共用此模块，避免重复定义。

MODEL_CONFIGS 格式:
    key → {'mode': 'local'|'api', ...}

yaml_name 后缀约定 (vlm_visual / vlm_hybrid):
    {model_key}[_{N}f]        -- N帧, e.g. default_2f / qwen25_7b_3f
    {model_key}_{vlm}_{trig}  -- hybrid专用, e.g. default_visual_kf
"""
import os
import difflib
import yaml

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

# DashScope API 模型白名单（alias -> 官方 model id）
# 注：alias 仅用于 tracker_param，可按需扩展。
API_MODEL_ALIASES = {
    'default': 'qwen3-vl-235b-a22b-instruct',
    # ---- Qwen3.5 / Qwen3.6 视觉理解 ----
    'qwen35_plus': 'qwen3.5-plus',
    'qwen35_flash': 'qwen3.5-flash',
    'qwen35_397b_a17b': 'qwen3.5-397b-a17b',
    'qwen35_122b_a10b': 'qwen3.5-122b-a10b',
    'qwen35_27b': 'qwen3.5-27b',
    'qwen35_35b_a3b': 'qwen3.5-35b-a3b',
    'qwen36_plus': 'qwen3.6-plus',
    # ---- Qwen3-VL ----
    'qwen3_vl_plus': 'qwen3-vl-plus',
    'qwen3_vl_flash': 'qwen3-vl-flash',
    'qwen3_vl_235b_a22b_instruct': 'qwen3-vl-235b-a22b-instruct',
    'qwen3_vl_235b_a22b_thinking': 'qwen3-vl-235b-a22b-thinking',
    'qwen3_vl_32b_instruct': 'qwen3-vl-32b-instruct',
    'qwen3_vl_30b_a3b_instruct': 'qwen3-vl-30b-a3b-instruct',
    'qwen3_vl_30b_a3b_thinking': 'qwen3-vl-30b-a3b-thinking',
    'qwen3_vl_8b_instruct': 'qwen3-vl-8b-instruct',
    'qwen3_vl_8b_thinking': 'qwen3-vl-8b-thinking',
    # ---- Qwen2.5-VL ----
    'qwen25_vl_72b': 'qwen2.5-vl-72b-instruct',
    'qwen25_vl_32b': 'qwen2.5-vl-32b-instruct',
    'qwen25_vl_7b': 'qwen2.5-vl-7b-instruct',
    'qwen25_vl_3b': 'qwen2.5-vl-3b-instruct',
    # ---- 历史兼容 ----
    'qwen_vl_max': 'qwen-vl-max',
    'qwen_vl_plus': 'qwen-vl-plus',
}

# 官方模型 id 白名单（允许直接写 api_<官方id>）
API_MODEL_IDS = sorted(set(API_MODEL_ALIASES.values()) | {
    # 你给出的主测集合
    'qwen3-vl-32b-instruct',
    'qwen3-vl-8b-instruct',
    'qwen3-vl-235b-a22b-instruct',
    'qwen2.5-vl-72b-instruct',
    'qwen2.5-vl-32b-instruct',
    'qwen2.5-vl-7b-instruct',
    'qwen2.5-vl-3b-instruct',
    'qwen3.5-27b',
    'qwen3.5-35b-a3b',
})

_API_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'


def _local_model_keys():
    return sorted([k for k, v in MODEL_CONFIGS.items() if v.get('mode') == 'local'])


def _raise_invalid_tracker_param(token: str, kind: str):
    if kind == 'api':
        alias_valid = sorted(API_MODEL_ALIASES.keys())
        id_valid = sorted(API_MODEL_IDS)
        valid = alias_valid + id_valid
        close = difflib.get_close_matches(token, valid, n=5, cutoff=0.45)
        hint = f", maybe: {', '.join(close)}" if close else ""
        raise ValueError(
            f"Invalid api model alias '{token}'. "
            f"Valid api aliases: {', '.join(alias_valid)}; "
            f"or official ids: {', '.join(id_valid)}{hint}"
        )

    if kind == 'local':
        valid = _local_model_keys()
        close = difflib.get_close_matches(token, valid, n=4, cutoff=0.45)
        hint = f", maybe: {', '.join(close)}" if close else ""
        raise ValueError(
            f"Invalid local model key '{token}'. "
            f"Valid local keys: {', '.join(valid)}{hint}"
        )

    valid = sorted(set(_local_model_keys() + ['default', 'api'] + [f'api_{k}' for k in API_MODEL_ALIASES]))
    close = difflib.get_close_matches(token, valid, n=4, cutoff=0.45)
    hint = f", maybe: {', '.join(close)}" if close else ""
    raise ValueError(
        f"Invalid tracker_param core token '{token}'. "
        f"Expect one of: default | api_<alias> | local_<model_key> | <local_model_key>{hint}"
    )


def _parse_model_key(token: str, strict: bool = False) -> str:
    """将 token 匹配到 MODEL_CONFIGS 的 key。strict=True 时非法直接报错。"""
    token = (token or '').strip().lower()

    # 1) 先精确匹配（优先兼容历史键，如 local_4b / local_8b）
    if token in MODEL_CONFIGS:
        return token

    # 2) 再做忽略下划线的模糊匹配
    for key in MODEL_CONFIGS:
        if key.replace('_', '') == token.replace('_', ''):
            return key

    # 3) 支持二级语义编码：
    #    local_<model_key>  -> 强制本地模式并解析模型
    #    api_<alias>        -> 统一走 api（alias 在 apply_vlm_config 校验）
    if token.startswith('local_'):
        inner = token[len('local_'):]
        if inner:
            if inner in MODEL_CONFIGS and MODEL_CONFIGS[inner].get('mode') == 'local':
                return inner
            for key in MODEL_CONFIGS:
                if MODEL_CONFIGS[key].get('mode') == 'local' and key.replace('_', '') == inner.replace('_', ''):
                    return key
        if strict:
            _raise_invalid_tracker_param(inner or token, kind='local')

    if token == 'api' or token.startswith('api_'):
        return 'api'

    if strict:
        if token.startswith('local_'):
            _raise_invalid_tracker_param(token[len('local_'):], kind='local')
        if token.startswith('api_'):
            _raise_invalid_tracker_param(token[len('api_'):], kind='api')
        _raise_invalid_tracker_param(token, kind='generic')

    return 'default'


def _split_core_and_extras(yaml_name: str):
    """解析 yaml_name，返回 (core_token, extras_dict)。"""
    tokens = (yaml_name or '').lower().split('_')
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

    core_token = '_'.join(remaining) if remaining else 'default'
    return core_token, extras


def parse_yaml_name(yaml_name: str):
    """
    解析 yaml_name，返回 (model_key, extras_dict)。

    vlm_visual 示例:
        'default'       → ('default', {})
        'default_2f'    → ('default', {'num_frames': 2})
        'qwen25_7b_3f'  → ('qwen25_7b', {'num_frames': 3})

    vlm_hybrid 示例:
        'default_visual_kf'       → ('default', {'vlm_mode': 'visual',    'trigger_mode': 'keyframe'})
        'default_cognitive_conf'  → ('default', {'vlm_mode': 'cognitive', 'trigger_mode': 'confidence'})
        'qwen3vl_4b_visual_hybrid'  → ('qwen3vl_4b', {'vlm_mode': 'visual',   'trigger_mode': 'hybrid'})
    """
    core_token, extras = _split_core_and_extras(yaml_name)
    model_key = _parse_model_key(core_token)
    return model_key, extras


def apply_vlm_config(params, yaml_name: str, strict: bool = True):
    """
    根据 yaml_name 将 VLM 模型参数写入 params 对象。

    Args:
        params:    TrackerParams 实例
        yaml_name: 配置名称字符串（支持后缀约定）

    Returns:
        extras (dict): 未被本函数消费的额外参数（如 num_frames, vlm_mode 等）
    """
    core_token, extras = _split_core_and_extras(yaml_name)
    token = core_token.strip().lower()
    model_key = _parse_model_key(token, strict=strict)

    # API 分支：api_<alias> 走显式白名单
    if token == 'api' or token.startswith('api_'):
        alias = 'default' if token == 'api' else token[len('api_'):].strip()
        if alias in API_MODEL_ALIASES:
            api_model = API_MODEL_ALIASES[alias]
        elif alias in API_MODEL_IDS:
            api_model = alias
        else:
            if strict:
                _raise_invalid_tracker_param(alias, kind='api')
            api_model = API_MODEL_ALIASES['default']

        params.mode = 'api'
        params.api_model = api_model
        params.api_base_url = _API_BASE_URL
        params.api_key = os.environ.get('DASHSCOPE_API_KEY', '')
        return extras

    # Local / default 分支
    config = MODEL_CONFIGS.get(model_key)
    if config is None:
        if strict:
            _raise_invalid_tracker_param(token, kind='generic')
        config = MODEL_CONFIGS['default']

    params.mode = config['mode']

    if params.mode == 'local':
        params.model_name = config.get('model_name', 'Qwen/Qwen2.5-VL-7B-Instruct')
        params.model_path = config.get('model_path', None)
    else:
        params.api_model = config.get('api_model', API_MODEL_ALIASES['default'])
        params.api_base_url = config.get('api_base_url', _API_BASE_URL)
        params.api_key = os.environ.get('DASHSCOPE_API_KEY', '')

    return extras


def load_vlm_experiment_defaults(tracker_name: str):
    """
    从 experiments/vlm/<tracker_name>.yaml 读取默认参数。

    用法目标：
        - tracker_name 决定“跟踪范式”的默认参数模板
        - tracker_param 再覆盖部署方式/模型选择
    """
    from lib.test.evaluation.environment import env_settings

    env = env_settings()
    yaml_file = os.path.join(env.prj_dir, 'experiments', 'vlm', f'{tracker_name}.yaml')

    if not os.path.isfile(yaml_file):
        return {}

    with open(yaml_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f'Invalid yaml format in {yaml_file}, expecting mapping.')

    return data


def apply_param_dict(params, cfg_dict: dict):
    """将字典中的键值直接写入 TrackerParams。"""
    for k, v in (cfg_dict or {}).items():
        setattr(params, k, v)

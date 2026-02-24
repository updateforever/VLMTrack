"""
Parameters for Qwen VLM Tracker - 基础两图跟踪

使用方法:
    python tracking/test.py qwen_vlm default --dataset lasot --debug 1
    python tracking/test.py qwen_vlm api --dataset lasot --debug 1
"""
from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings


# 模型配置字典
MODEL_CONFIGS = {
    # Local models
    'local_4b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen3-VL-4B-Instruct',
    },
    'local_8b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen3-VL-8B-Instruct',
    },
    
    # API models
    'api': {
        'mode': 'api',
        'api_model': 'qwen3-vl-235b-a22b-instruct',
        'api_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    },
    
    # Default
    'default': {
        'mode': 'api',
        'api_model': 'qwen3-vl-235b-a22b-instruct',
        'api_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    },
}


def parameters(yaml_name: str = "default"):
    """
    基础VLM跟踪器参数
    
    Args:
        yaml_name: 配置名称，可选:
            - 'default' / 'api': API模式
            - 'local_4b': 本地4B模型
            - 'local_8b': 本地8B模型
    """
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    
    # 获取配置
    config = MODEL_CONFIGS.get(yaml_name.lower(), MODEL_CONFIGS['default'])
    params.mode = config.get('mode', 'api')
    
    # VLM配置
    if params.mode == 'local':
        params.model_name = config.get('model_name', 'Qwen/Qwen3-VL-4B-Instruct')
        params.model_path = config.get('model_path', None)
        print(f"[QwenVLM] Local: {params.model_name}")
    else:
        params.api_model = config.get('api_model', 'qwen3-vl-235b-a22b-instruct')
        params.api_base_url = config.get('api_base_url', 
                                         'https://dashscope.aliyuncs.com/compatible-mode/v1')
        params.api_key = os.environ.get('DASHSCOPE_API_KEY', '')
        print(f"[QwenVLM] API: {params.api_model}")
    
    # Prompt配置
    params.prompt_name = 'two_image'  # 使用两图跟踪prompt
    
    # Debug配置
    params.debug = 0  # 0=无输出, 1=打印, 2=保存图片, 3=保存+显示
    
    # 关键帧稀疏推理配置
    params.use_keyframe = True  # 默认关闭，需要时设置use_keyframe=True
    env = env_settings()
    # 新版配置结构：支持模型选择和阈值配置
    # 如果env中有配置，使用env的配置；否则使用默认配置
    if hasattr(env, 'keyframe_root') and isinstance(env.keyframe_root, dict):
        params.keyframe_root = env.keyframe_root
    else:
        # 默认配置示例
        params.keyframe_root = {
            'root': '/data/DATASETS_PUBLIC/SOIBench_f',
            'model': 'scene_changes_clip',  # 或 scene_changes_resnet
            'threshold': 'top_10',          # 或 top_30, top_80
        }
    
    # 兼容性参数（框架需要）
    params.checkpoint = None
    params.save_all_boxes = False  # VLM tracker不需要保存所有框
    
    return params

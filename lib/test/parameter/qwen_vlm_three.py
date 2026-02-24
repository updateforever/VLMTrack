"""
Parameters for Qwen VLM Three-Image Tracker - 三图跟踪

使用方法:
    python tracking/test.py qwen_vlm_three default --dataset lasot --debug 1
    python tracking/test.py qwen_vlm_three api --dataset lasot --debug 1
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
    三图VLM跟踪器参数
    
    跟踪范式: 初始帧+绿框 + 上一帧+蓝框 + 当前帧
    
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
        print(f"[QwenThree] Local: {params.model_name}")
    else:
        params.api_model = config.get('api_model', 'qwen3-vl-235b-a22b-instruct')
        params.api_base_url = config.get('api_base_url', 
                                         'https://dashscope.aliyuncs.com/compatible-mode/v1')
        params.api_key = os.environ.get('DASHSCOPE_API_KEY', '')
        print(f"[QwenThree] API: {params.api_model}")
    
    # Prompt配置
    params.prompt_name = 'three_image'  # 使用三图跟踪prompt
    
    # Debug配置
    params.debug = 0  # 0=无输出, 1=打印, 2=保存图片, 3=保存+显示
    
    # 关键帧稀疏推理配置
    params.use_keyframe = False  # 默认关闭
    env = env_settings()
    # 新版配置结构
    if hasattr(env, 'keyframe_root') and isinstance(env.keyframe_root, dict):
        params.keyframe_root = env.keyframe_root
    else:
        params.keyframe_root = {
            'root': '/data/DATASETS_PUBLIC/SOIBench_f',
            'model': 'scene_changes_clip',
            'threshold': 'top_10',
        }
    
    # 兼容性参数（框架需要）
    params.checkpoint = None
    params.save_all_boxes = False  # VLM tracker不需要保存所有框
    
    return params

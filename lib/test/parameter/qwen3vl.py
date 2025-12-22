"""
Parameters for Qwen3VL Tracker
支持本地推理和API推理两种模式

使用方法:
    # 本地推理 (单线程)
    python tracking/test.py qwen3vl qwen3vl_4b --dataset tnl2k --threads 0
    
    # API推理 (支持多线程!)
    python tracking/test.py qwen3vl qwen3vl_api --dataset tnl2k --threads 4
"""
from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings


# =========== 模型配置 ===========
MODEL_CONFIGS = {
    # Qwen3-VL Dense Models (推荐)
    'qwen3vl_4b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen3-VL-4B-Instruct',
    },
    'qwen3vl_8b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen3-VL-8B-Instruct',
    },
    'qwen3vl_32b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen3-VL-32B-Instruct',
    },
    
    # Qwen2.5-VL Models
    'qwen25vl_3b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen2.5-VL-3B-Instruct',
    },
    'qwen25vl_7b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen2.5-VL-7B-Instruct',
    },
    'qwen25vl_32b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen2.5-VL-32B-Instruct',
    },
    'qwen25vl_72b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen2.5-VL-72B-Instruct',
    },
    
    # API Models (支持多线程!)
    'qwen3vl_api': {
        'mode': 'api',
        'api_model': 'qwen3-vl-235b-a22b-instruct',
        'api_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    },
    'qwen25vl_api': {
        'mode': 'api',
        'api_model': 'qwen-vl-max',
        'api_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    },
}


def parameters(yaml_name: str = "qwen3vl_4b"):
    """
    Get parameters for Qwen3VL tracker
    
    Args:
        yaml_name: 配置名称
            本地模型: 'qwen3vl_4b', 'qwen3vl_8b', 'qwen3vl_32b', 
                     'qwen25vl_3b', 'qwen25vl_7b', ...
            API模型: 'qwen3vl_api', 'qwen25vl_api' (支持多线程)
    """
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    
    # 获取配置
    config = MODEL_CONFIGS.get(yaml_name.lower(), MODEL_CONFIGS['qwen3vl_4b'])
    
    # 设置模式
    params.mode = config.get('mode', 'local')
    
    if params.mode == 'local':
        # 本地模型配置
        params.model_name = config.get('model_name', 'Qwen/Qwen3-VL-4B-Instruct')
        params.model_path = config.get('model_path', None)  # 可选本地路径
        print(f"[Qwen3VL] Local mode: {params.model_name}")
    else:
        # API配置
        params.api_model = config.get('api_model', 'qwen3-vl-235b-a22b-instruct')
        params.api_base_url = config.get('api_base_url', 
                                         'https://dashscope.aliyuncs.com/compatible-mode/v1')
        # API key从环境变量获取
        params.api_key = os.environ.get('DASHSCOPE_API_KEY', '')
        print(f"[Qwen3VL] API mode: {params.api_model}")
        if not params.api_key:
            print("[Qwen3VL] ⚠️ Warning: DASHSCOPE_API_KEY not set!")
    
    # =========== 兼容性设置 ===========
    params.template_factor = 2.0
    params.template_size = 112
    params.search_factor = 4.0
    params.search_size = 224
    
    # =========== Debug设置 ===========
    # 0: 无输出, 1: 打印信息, 2: 保存可视化, 3: 实时显示
    params.debug = 0
    
    params.save_all_boxes = False
    params.checkpoint = None
    
    return params

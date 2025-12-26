"""
Parameters for Qwen3VL Hybrid Tracker
混合模式: 三图 + 记忆库

使用方法:
    python tracking/test.py qwen3vl_hybrid qwen3vl_hybrid_api --dataset tnl2k --threads 0 --debug 1
"""
from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings


# =========== 模型配置 ===========
MODEL_CONFIGS = {
    # Qwen3-VL Dense Models
    'qwen3vl_hybrid_4b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen3-VL-4B-Instruct',
    },
    'qwen3vl_hybrid_8b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen3-VL-8B-Instruct',
    },
    'qwen3vl_hybrid_32b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen3-VL-32B-Instruct',
    },
    
    # Qwen2.5-VL Models
    'qwen25vl_hybrid_3b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen2.5-VL-3B-Instruct',
    },
    'qwen25vl_hybrid_7b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen2.5-VL-7B-Instruct',
    },
    
    # API Models
    'qwen3vl_hybrid_api': {
        'mode': 'api',
        'api_model': 'qwen3-vl-235b-a22b-instruct',
        'api_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    },
    'qwen25vl_hybrid_api': {
        'mode': 'api',
        'api_model': 'qwen-vl-max',
        'api_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    },
}


def parameters(yaml_name: str = "qwen3vl_hybrid_api"):
    """
    混合跟踪参数
    
    Args:
        yaml_name: 配置名称
            本地模型: 'qwen3vl_hybrid_4b', 'qwen3vl_hybrid_8b', ...
            API模型: 'qwen3vl_hybrid_api', 'qwen25vl_hybrid_api'
    """
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    
    # 获取配置
    config = MODEL_CONFIGS.get(yaml_name.lower(), MODEL_CONFIGS['qwen3vl_hybrid_api'])
    
    # 设置模式
    params.mode = config.get('mode', 'local')
    
    if params.mode == 'local':
        params.model_name = config.get('model_name', 'Qwen/Qwen3-VL-4B-Instruct')
        params.model_path = config.get('model_path', None)
        print(f"[Hybrid] Local mode: {params.model_name}")
    else:
        params.api_model = config.get('api_model', 'qwen3-vl-plus-2025-09-23')
        params.api_base_url = config.get('api_base_url', 
                                         'http://10.128.202.100:3010/v1')
        params.api_key = os.environ.get('DASHSCOPE_API_KEY', '')
        print(f"[Hybrid] API mode: {params.api_model}")
        if not params.api_key:
            print("[Hybrid] ⚠️ Warning: DASHSCOPE_API_KEY not set!")
    
    # =========== 记忆库配置 ===========
    params.memory_update_interval = 10  # 每10帧更新记忆
    params.memory_confidence_threshold = 0.8
    
    # =========== 兼容性设置 ===========
    params.template_factor = 2.0
    params.template_size = 112
    params.search_factor = 4.0
    params.search_size = 224
    
    # =========== Debug设置 ===========
    params.debug = 0
    
    # =========== 关键帧跟踪 ===========
    params.use_keyframe = True
    params.keyframe_root = "/home/member/data2/wyp/SOT/VLMTrack/scene_changes_clip/tnl2k_test_scene_changes_clip"
    
    params.save_all_boxes = False
    params.checkpoint = None
    
    return params

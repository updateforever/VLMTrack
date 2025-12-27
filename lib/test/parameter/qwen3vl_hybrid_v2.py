"""
Parameters for Qwen3VL Hybrid V2 Tracker
混合模式V2: 三图 + 记忆库V2 (一次VLM调用)

使用方法:
    python tracking/test.py qwen3vl_hybrid_v2 qwen3vl_hybrid_v2_api --dataset tnl2k --threads 0 --debug 1
"""
from lib.test.utils import TrackerParams
import os


MODEL_CONFIGS = {
    # Local models
    'qwen3vl_hybrid_v2_4b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen3-VL-4B-Instruct',
    },
    'qwen3vl_hybrid_v2_8b': {
        'mode': 'local',
        'model_name': 'Qwen/Qwen3-VL-8B-Instruct',
    },
    
    # API models
    'qwen3vl_hybrid_v2_api': {
        'mode': 'api',
        'api_model': 'qwen3-vl-235b-a22b-instruct',
        'api_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    },
}


def parameters(yaml_name: str = "qwen3vl_hybrid_v2_api"):
    """Hybrid V2参数"""
    params = TrackerParams()
    
    config = MODEL_CONFIGS.get(yaml_name.lower(), MODEL_CONFIGS['qwen3vl_hybrid_v2_api'])
    
    params.mode = config.get('mode', 'local')
    
    if params.mode == 'local':
        params.model_name = config.get('model_name', 'Qwen/Qwen3-VL-4B-Instruct')
        params.model_path = config.get('model_path', None)
        print(f"[HybridV2] Local: {params.model_name}")
    else:
        params.api_model = config.get('api_model', 'qwen3-vl-235b-a22b-instruct')
        params.api_base_url = config.get('api_base_url', 
                                         'https://dashscope.aliyuncs.com/compatible-mode/v1')
        params.api_key = os.environ.get('DASHSCOPE_API_KEY', '')
        print(f"[HybridV2] API: {params.api_model}")
    
    # 兼容性
    params.template_factor = 2.0
    params.template_size = 112
    params.search_factor = 4.0
    params.search_size = 224
    
    # Debug
    params.debug = 0
    
    # 关键帧
    params.use_keyframe = True
    params.keyframe_root = "/home/member/data2/wyp/SOT/VLMTrack/scene_changes_clip/tnl2k_test_scene_changes_clip"
    
    params.save_all_boxes = False
    params.checkpoint = None
    
    return params

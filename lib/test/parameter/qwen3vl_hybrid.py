"""
Parameters for Qwen3VL Hybrid Tracker
混合模式: 三图 + 记忆库

使用方法:
    python tracking/test.py qwen3vl_hybrid qwen3vl_hybrid_api --dataset tnl2k --threads 0 --debug 1
"""
from lib.test.utils import TrackerParams
import os


def parameters(yaml_name: str = "qwen3vl_hybrid_api"):
    """混合跟踪参数"""
    params = TrackerParams()
    
    # API模式
    if 'api' in yaml_name.lower():
        params.mode = 'api'
        params.api_model = 'qwen3-vl-plus-2025-09-23'
        params.api_base_url = 'http://10.128.202.100:3010/v1'
        params.api_key = os.environ.get('DASHSCOPE_API_KEY', '')
    else:
        params.mode = 'local'
        params.model_name = 'Qwen/Qwen3-VL-4B-Instruct'
    
    # 记忆库配置
    params.memory_update_interval = 10
    
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

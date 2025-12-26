"""
Parameters for Qwen3VL Memory Bank Tracker
记忆库跟踪模式: VLM维护目标的语义记忆

使用方法:
    python tracking/test.py qwen3vl_memory qwen3vl_memory_api --dataset tnl2k --threads 0 --debug 1
"""
from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings


def parameters(yaml_name: str = "qwen3vl_memory_api"):
    """
    记忆库跟踪参数
    
    Args:
        yaml_name: 'qwen3vl_memory_api' (推荐) 或 'qwen3vl_memory_local'
    """
    params = TrackerParams()
    
    # 模式选择
    if 'api' in yaml_name.lower():
        params.mode = 'api'
        params.api_model = 'qwen3-vl-plus-2025-09-23'
        params.api_base_url = 'http://10.128.202.100:3010/v1'
        params.api_key = os.environ.get('DASHSCOPE_API_KEY', '')
        print(f"[Memory] API mode: {params.api_model}")
    else:
        params.mode = 'local'
        params.model_name = 'Qwen/Qwen3-VL-4B-Instruct'
        params.model_path = None
        print(f"[Memory] Local mode: {params.model_name}")
    
    # 记忆库配置
    params.memory_update_interval = 10  # 每10帧更新记忆
    params.memory_confidence_threshold = 0.8  # 记忆置信度阈值
    
    # 兼容性设置
    params.template_factor = 2.0
    params.template_size = 112
    params.search_factor = 4.0
    params.search_size = 224
    
    # Debug (由test.py --debug控制)
    params.debug = 0
    
    # 关键帧跟踪
    params.use_keyframe = True
    params.keyframe_root = "/home/member/data2/wyp/SOT/VLMTrack/scene_changes_clip/tnl2k_test_scene_changes_clip"
    
    params.save_all_boxes = False
    params.checkpoint = None
    
    return params

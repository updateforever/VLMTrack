"""
Parameters for QwenVLMHybrid - 混合跟踪器

yaml_name 后缀约定:
    {model}_{vlm_mode}_{trigger}

    model:   default | api | local_4b | local_8b
    vlm:     visual | cognitive
    trigger: kf(keyframe) | conf(confidence) | hybrid

示例:
    default_visual_kf        → API, Visual VLM, 关键帧触发
    default_cognitive_kf     → API, Cognitive VLM, 关键帧触发
    default_visual_conf      → API, Visual VLM, 置信度触发
    default_cognitive_hybrid → API, Cognitive VLM, 两者取其一
    local_4b_visual_kf       → 本地4B, Visual VLM, 关键帧触发

使用方法:
    python tracking/test.py qwen_vlm_hybrid default_visual_kf   --dataset_name lasot
    python tracking/test.py qwen_vlm_hybrid default_cognitive_conf --dataset_name lasot
"""
from lib.test.utils import TrackerParams
from lib.test.evaluation.environment import env_settings
from lib.test.parameter.vlm_common import apply_vlm_config
from lib.config.sutrack.config import cfg, update_config_from_file
import os


# SUTrack 可用配置映射
# key → yaml 文件名（experiments/sutrack/目录下）
SUTRACK_CONFIGS = {
    'b224': 'sutrack_b224',
    'b384': 'sutrack_b384',
    'l224': 'sutrack_l224',
    'l384': 'sutrack_l384',
    # 默认
    'default': 'sutrack_b384',
}


def parameters(yaml_name: str = "default_visual_kf"):
    params = TrackerParams()
    env = env_settings()

    # ---- 解析 yaml_name: VLM 模型 + vlm_mode + trigger_mode ----
    extras = apply_vlm_config(params, yaml_name)

    params.vlm_mode = extras.get('vlm_mode', 'visual')
    params.trigger_mode = extras.get('trigger_mode', 'keyframe')
    params.num_frames = extras.get('num_frames', 2)
    params.prompt_name = 'two_image' if params.num_frames == 2 else 'three_image'

    # ---- SUTrack 配置 ----
    sutrack_key = getattr(params, 'sutrack_config', 'default')
    sutrack_yaml = SUTRACK_CONFIGS.get(sutrack_key, SUTRACK_CONFIGS['default'])
    yaml_file = os.path.join(env.prj_dir, f'experiments/sutrack/{sutrack_yaml}.yaml')
    update_config_from_file(yaml_file)
    params.cfg = cfg

    # SUTrack 搜索/模板参数（从 cfg 读取）
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.checkpoint = os.path.join(
        env.save_dir,
        f"checkpoints/train/sutrack/{sutrack_yaml}/SUTRACK_ep{cfg.TEST.EPOCH:04d}.pth.tar"
    )

    # ---- VLM Cognitive 模式专用 ----
    params.track_prompt = 'cognitive'
    params.init_prompt = 'init_memory'

    # ---- 置信度触发阈值 ----
    params.conf_threshold = 0.3  # SUTrack 置信度低于此值时触发 VLM

    # ---- 调试 ----
    params.debug = 0  # 0=静默, 1=控制台, 2=保存可视化

    # ---- 稀疏关键帧 ----
    params.use_keyframe = True
    params.keyframe_root = getattr(env, 'keyframe_root', '')

    # ---- 框架兼容 ----
    params.save_all_boxes = False

    return params

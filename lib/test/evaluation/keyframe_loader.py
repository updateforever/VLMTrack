"""
关键帧索引加载模块（简化版）

固定路径格式:
    {keyframe_root}/{dataset}/{split}/{seq_name}.json

keyframe_root 由 local.py 中的 env.keyframe_root 设置，
路径中已包含检测模型和阈值层级，例如:
    /data/DATASETS_PUBLIC/SOIBench/KeyFrame/scene_changes_resnet/top_10
"""
import json
import warnings
from pathlib import Path
from typing import Optional, Set


# dataset_name → (dataset目录, split) 映射表
# 新增数据集时只需在此添加一行即可
_DATASET_MAP = {
    # LaSOT
    'lasot':        ('lasot', 'test'),
    'lasot_test':   ('lasot', 'test'),
    'lasot_val':    ('lasot', 'val'),
    'lasot_train':  ('lasot', 'train'),
    # VideoCube/MGIT（关键帧目录下使用 mgit/）
    'videocube':           ('mgit', 'test'),
    'videocube_test':      ('mgit', 'test'),
    'videocube_val':       ('mgit', 'val'),
    'videocube_test_tiny': ('mgit', 'test'),
    'videocube_val_tiny':  ('mgit', 'val'),
    # TNL2K
    'tnl2k':        ('tnl2k', 'test'),
    'tnl2k_test':   ('tnl2k', 'test'),
    'tnl2k_val':    ('tnl2k', 'val'),
    # MGIT（直接使用 mgit 名称时）
    'mgit':         ('mgit', 'test'),
    'mgit_test':    ('mgit', 'test'),
    'mgit_val':     ('mgit', 'val'),
}


def load_keyframe_indices(
    dataset_name: str,
    seq_name: str,
    keyframe_root: str,
) -> Optional[Set[int]]:
    """
    加载关键帧索引。

    Args:
        dataset_name: 数据集名称，如 'lasot', 'videocube_val', 'tnl2k'
        seq_name:     序列名称，如 'airplane-1'
        keyframe_root: 关键帧根目录（已包含 model 和 threshold 层级），
                       如 '/data/DATASETS_PUBLIC/SOIBench/KeyFrame/scene_changes_resnet/top_10'

    Returns:
        关键帧帧号集合（0-indexed），找不到或出错时返回 None。

    路径格式:
        {keyframe_root}/{dataset}/{split}/{seq_name}.json
        或
        {keyframe_root}/{dataset}/{split}/frame_{seq_name}.json
    示例:
        /data/.../top_10/lasot/test/airplane-1.json
        /data/.../top_10/mgit/val/seq001.json
    """
    if not keyframe_root:
        return None

    entry = _DATASET_MAP.get(dataset_name.lower())
    if entry is None:
        warnings.warn(
            f"[KeyframeLoader] Unknown dataset_name: '{dataset_name}'. "
            f"Please add it to _DATASET_MAP in keyframe_loader.py."
        )
        return None

    dataset_dir, split = entry
    base_dir = Path(keyframe_root) / dataset_dir / split

    # 兼容两种命名:
    # 1) {seq_name}.json
    # 2) frame_{seq_name}.json
    seq_core = seq_name[len("frame_"):] if seq_name.startswith("frame_") else seq_name
    candidate_names = [f"{seq_core}.json", f"frame_{seq_core}.json"]

    candidate_paths = []
    seen = set()
    for name in candidate_names:
        path = base_dir / name
        if path not in seen:
            seen.add(path)
            candidate_paths.append(path)

    index_path = None
    for path in candidate_paths:
        if path.exists():
            index_path = path
            break

    if index_path is None:
        tried = ", ".join(str(p.name) for p in candidate_paths)
        warnings.warn(f"[KeyframeLoader] Index file not found in {base_dir}. Tried: {tried}")
        return None

    return _parse_index_file(index_path, seq_name)


def _parse_index_file(path: Path, seq_name: str) -> Optional[Set[int]]:
    """
    解析关键帧索引文件。

    支持两种格式:
        1. {"key_frames": [0, 8, 17, ...]}   ← 标准格式
        2. [0, 8, 17, ...]                   ← 纯数组格式
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict) and 'key_frames' in data:
            indices = set(data['key_frames'])
        elif isinstance(data, list):
            indices = set(data)
        else:
            warnings.warn(f"[KeyframeLoader] Unrecognized format in {path}")
            return None

        return indices


    except json.JSONDecodeError as e:
        warnings.warn(f"[KeyframeLoader] JSON parse error in {path}: {e}")
        return None
    except Exception as e:
        warnings.warn(f"[KeyframeLoader] Error reading {path}: {e}")
        return None

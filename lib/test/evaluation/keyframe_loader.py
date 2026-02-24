"""
关键帧索引加载模块

用于VLM稀疏推理的关键帧索引管理。
根据数据集名和序列名加载预先计算的关键帧索引。
"""
import json
import os
from pathlib import Path
from typing import Optional, Set, List
import warnings


class KeyframeIndexLoader:
    """
    关键帧索引加载器
    
    支持的索引文件格式:
    1. demo.json格式: 单个序列的索引文件，包含key_frames字段
    2. dataset_index.json格式: 多个序列的索引文件（未来扩展）
    """
    
    def __init__(self, keyframe_config=None):
        """
        初始化关键帧索引加载器
        
        Args:
            keyframe_config: 关键帧配置，支持多种格式：
                - None: 不使用关键帧
                - str: 旧版单个根目录路径（向后兼容）
                - dict: 完整配置，包含：
                    {
                        'root': '/data/DATASETS_PUBLIC/SOIBench_f',
                        'model': 'scene_changes_clip',      # 或 scene_changes_resnet
                        'threshold': 'top_10',              # 或 top_30, top_80
                    }
        """
        if isinstance(keyframe_config, dict):
            self.root = Path(keyframe_config.get('root', ''))
            self.model = keyframe_config.get('model', 'scene_changes_clip')
            self.threshold = keyframe_config.get('threshold', 'top_10')
            self.is_new_structure = True
        elif keyframe_config:
            # 向后兼容：旧版字符串路径
            self.root = Path(keyframe_config)
            self.model = None
            self.threshold = None
            self.is_new_structure = False
        else:
            self.root = None
            self.model = None
            self.threshold = None
            self.is_new_structure = False
        
        self._cache = {}  # 缓存已加载的索引
        
    def load_keyframe_indices(
        self, 
        dataset_name: str, 
        seq_name: str,
        keyframe_config=None
    ) -> Optional[Set[int]]:
        """
        根据数据集名和序列名加载关键帧索引
        
        Args:
            dataset_name: 数据集名称，格式如 'lasot', 'lasot_test', 'videocube_val'
            seq_name: 序列名称，如 'airplane-1', 'bear-1'
            keyframe_config: 关键帧配置（可选，如果init时未指定）
            
        Returns:
            关键帧索引集合，如果未找到则返回None
            
        新版路径结构:
            {root}/{model}/{threshold}/{dataset}/{split}/{seq_name}.jsonl
            例如: /data/.../SOIBench_f/scene_changes_clip/top_10/lasot/test/airplane-1.jsonl
            
        从dataset_name解析dataset和split:
            - 'lasot' → dataset='lasot', split='test' (默认)
            - 'lasot_test' → dataset='lasot', split='test'
            - 'videocube_val' → dataset='videocube', split='val'
        """
        # 确定使用哪个配置
        if keyframe_config:
            if isinstance(keyframe_config, dict):
                root = Path(keyframe_config['root'])
                model = keyframe_config.get('model', 'scene_changes_clip')
                threshold = keyframe_config.get('threshold', 'top_10')
                is_new = True
            else:
                root = Path(keyframe_config)
                model = None
                threshold = None
                is_new = False
        else:
            root = self.root
            model = self.model
            threshold = self.threshold
            is_new = self.is_new_structure
        
        if not root or not root.exists():
            return None
        
        # 检查缓存
        cache_key = f"{dataset_name}:{seq_name}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 根据路径结构选择查找方式
        if is_new and model and threshold:
            # 新版结构
            candidates = self._get_new_structure_paths(root, model, threshold, dataset_name, seq_name)
        else:
            # 旧版结构（向后兼容）
            candidates = self._get_legacy_paths(root, dataset_name, seq_name)
        
        # 尝试加载
        for path in candidates:
            if path.exists():
                indices = self._parse_index_file(path, seq_name)
                if indices is not None:
                    self._cache[cache_key] = indices
                    return indices
        
        # 未找到索引文件
        warnings.warn(
            f"[KeyframeLoader] No index file found for {dataset_name}/{seq_name}. "
            f"Searched paths: {[str(p) for p in candidates[:3]]}"
        )
        return None
    
    def _parse_dataset_name(self, dataset_name: str) -> tuple:
        """
        从dataset_name解析出dataset和split
        
        Args:
            dataset_name: 如 'lasot', 'lasot_test', 'videocube_val_tiny'
            
        Returns:
            (dataset, split) 元组
            
        示例:
            'lasot' → ('lasot', 'test')
            'lasot_test' → ('lasot', 'test')
            'videocube_val' → ('mgit', 'val')
            'videocube_val_tiny' → ('mgit', 'val')
            'videocube_test_tiny' → ('mgit', 'test')
            'tnl2k' → ('tnl2k', 'test')
        """
        name_lower = dataset_name.lower()
        
        # 先去掉version后缀（如 _tiny）
        version_suffixes = ['_tiny']
        for version_suffix in version_suffixes:
            if name_lower.endswith(version_suffix):
                name_lower = name_lower[:-len(version_suffix)]
                break
        
        # 解析split后缀
        splits = ['_test', '_val', '_train']
        dataset = name_lower
        split = 'test'  # 默认
        
        for split_suffix in splits:
            if name_lower.endswith(split_suffix):
                dataset = name_lower[:-len(split_suffix)]
                split = split_suffix[1:]  # 去掉前导下划线
                break
        
        # 特殊映射：videocube → mgit（关键帧路径中使用mgit）
        if dataset == 'videocube':
            dataset = 'mgit'
        
        return dataset, split
    
    def _get_new_structure_paths(
        self,
        root: Path,
        model: str,
        threshold: str,
        dataset_name: str,
        seq_name: str
    ) -> List[Path]:
        """
        获取新版路径结构的候选路径
        
        路径格式: {root}/{model}/{threshold}/{dataset}/{split}/{seq_name}.jsonl
        
        示例:
            /data/SOIBench_f/scene_changes_clip/top_10/lasot/test/airplane-1.jsonl
        """
        candidates = []
        
        # 解析dataset和split
        dataset, split = self._parse_dataset_name(dataset_name)
        
        # 标准路径: {root}/{model}/{threshold}/{dataset}/{split}/{seq_name}.jsonl
        base_path = root / model / threshold / dataset / split
        candidates.append(base_path / f"{seq_name}.jsonl")
        candidates.append(base_path / f"{seq_name}.json")
        candidates.append(base_path / f"frame_{seq_name}.json")
        
        # # 如果没有明确的split后缀，也尝试其他split
        # if not any(dataset_name.lower().endswith(s) for s in ['_test', '_val', '_train']):
        #     for other_split in ['val', 'train']:
        #         other_path = root / model / threshold / dataset / other_split
        #         candidates.append(other_path / f"{seq_name}.jsonl")
        #         candidates.append(other_path / f"{seq_name}.json")
        
        return candidates
    
    def _get_legacy_paths(
        self,
        root: Path,
        dataset_name: str,
        seq_name: str
    ) -> List[Path]:
        """
        获取旧版路径结构的候选路径（向后兼容）
        
        路径优先级:
        1. dataset/seq_name.jsonl
        2. dataset/seq_name.json
        3. dataset/category/seq_name.jsonl
        4. seq_name.jsonl
        """
        candidates = []
        
        # 解析dataset (去掉split后缀)
        dataset, _ = self._parse_dataset_name(dataset_name)
        
        # 标准路径
        candidates.append(root / dataset / f"{seq_name}.jsonl")
        candidates.append(root / dataset / f"{seq_name}.json")
        
        # 带类别路径 (如 lasot/airplane/airplane-1.json)
        if '-' in seq_name:
            category = seq_name.rsplit('-', 1)[0]
            candidates.append(root / dataset / category / f"{seq_name}.jsonl")
            candidates.append(root / dataset / category / f"{seq_name}.json")
        
        # 根目录直接查找
        candidates.append(root / f"{seq_name}.jsonl")
        candidates.append(root / f"{seq_name}.json")
        
        return candidates
    
    def _parse_index_file(self, path: Path, seq_name: str) -> Optional[Set[int]]:
        """
        解析索引文件
        
        支持的格式:
        1. demo.json格式:
           {
               "sequence": "frame_005",
               "key_frames": [0, 8, 17, 25, ...]
           }
        
        2. 简化格式:
           {
               "seq_name": {
                   "key_frames": [0, 10, 20, ...]
               }
           }
        
        3. 纯数组格式:
           [0, 10, 20, 30, ...]
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 格式1: demo.json格式 (单序列)
            if isinstance(data, dict) and 'key_frames' in data:
                key_frames = data['key_frames']
                if isinstance(key_frames, list):
                    indices = set(key_frames)
                    print(f"[KeyframeLoader] Loaded {len(indices)} keyframes for {seq_name} from {path.name}")
                    return indices
            
            # 格式2: 多序列索引文件
            if isinstance(data, dict) and seq_name in data:
                seq_data = data[seq_name]
                if isinstance(seq_data, dict) and 'key_frames' in seq_data:
                    key_frames = seq_data['key_frames']
                    if isinstance(key_frames, list):
                        indices = set(key_frames)
                        print(f"[KeyframeLoader] Loaded {len(indices)} keyframes for {seq_name} from {path.name}")
                        return indices
            
            # 格式3: 纯数组格式
            if isinstance(data, list):
                indices = set(data)
                print(f"[KeyframeLoader] Loaded {len(indices)} keyframes for {seq_name} from {path.name}")
                return indices
            
            warnings.warn(f"[KeyframeLoader] Unrecognized format in {path}")
            return None
            
        except json.JSONDecodeError as e:
            warnings.warn(f"[KeyframeLoader] JSON parse error in {path}: {e}")
            return None
        except Exception as e:
            warnings.warn(f"[KeyframeLoader] Error reading {path}: {e}")
            return None
    
    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()
    
    def preload_dataset(self, dataset_name: str) -> int:
        """
        预加载整个数据集的索引文件
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            成功加载的序列数量
        """
        if not self.index_root:
            return 0
        
        dataset_dir = self.index_root / dataset_name.lower()
        if not dataset_dir.exists():
            return 0
        
        count = 0
        for json_file in dataset_dir.rglob("*.json"):
            seq_name = json_file.stem
            indices = self._parse_index_file(json_file, seq_name)
            if indices:
                cache_key = f"{dataset_name}:{seq_name}"
                self._cache[cache_key] = indices
                count += 1
        
        print(f"[KeyframeLoader] Preloaded {count} sequences for {dataset_name}")
        return count


# 全局单例实例
_global_loader: Optional[KeyframeIndexLoader] = None


def get_keyframe_loader(keyframe_config=None) -> KeyframeIndexLoader:
    """
    获取全局关键帧加载器实例（单例模式）
    
    Args:
        keyframe_config: 关键帧配置（首次调用时设置）
            - None: 不使用关键帧
            - str: 旧版根目录路径
            - dict: 新版配置 {'root': ..., 'model': ..., 'threshold': ...}
        
    Returns:
        KeyframeIndexLoader实例
    """
    global _global_loader
    
    if _global_loader is None:
        _global_loader = KeyframeIndexLoader(keyframe_config)
    elif keyframe_config is not None:
        # 如果配置改变，重新创建
        _global_loader = KeyframeIndexLoader(keyframe_config)
    
    return _global_loader


def load_keyframe_indices(
    dataset_name: str,
    seq_name: str,
    keyframe_config=None
) -> Optional[Set[int]]:
    """
    便捷函数：加载关键帧索引
    
    Args:
        dataset_name: 数据集名称，如 'lasot', 'lasot_test', 'videocube_val'
        seq_name: 序列名称
        keyframe_config: 关键帧配置（str或dict）
        
    Returns:
        关键帧索引集合
    """
    loader = get_keyframe_loader(keyframe_config)
    return loader.load_keyframe_indices(dataset_name, seq_name, keyframe_config)

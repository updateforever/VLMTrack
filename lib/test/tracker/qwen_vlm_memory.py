"""
Qwen VLM Tracker - 记忆库版本

三图 + 记忆库跟踪：使用语义记忆辅助跟踪

特点:
- 初始帧生成语义记忆
- 跟踪时使用记忆 + 上一帧运动线索
- 一次VLM调用同时输出bbox和更新的记忆
"""
from lib.test.tracker.basetracker import BaseTracker
from lib.test.evaluation.environment import env_settings
import cv2
import numpy as np
import os
import re
import json
from typing import List, Dict

from lib.test.tracker.vlm_engine import VLMEngine
from lib.test.tracker.vlm_utils import (
    parse_bbox_from_text, xyxy_to_xywh, draw_bbox,
    parse_memory_state, dict_to_str
)
from lib.test.tracker.prompts import get_prompt


class QwenVLMMemory(BaseTracker):
    """
    记忆库VLM跟踪器
    
    跟踪范式:
    1. 初始化: 从初始帧生成语义记忆
    2. 跟踪: 使用记忆 + 上一帧 → 预测bbox + 更新记忆
    """
    
    def __init__(self, params, dataset_name):
        super().__init__(params)
        self.params = params
        self.dataset_name = dataset_name
        
        # VLM引擎
        self.vlm = VLMEngine(params)
        
        # Tracker状态
        self.state = None
        self.prev_image = None
        self.prev_bbox = None
        self.language_description = None
        
        # 记忆库
        self.memory = {
            "appearance": "",
            "motion": "",
            "context": "",
            "last_update": 0
        }
        
        # 帧计数
        self.frame_id = 0
        self.seq_name = None
        
        # 从params读取Prompt配置
        self.track_prompt = getattr(params, 'track_prompt', 'memory_bank')
        self.init_prompt = getattr(params, 'init_prompt', 'init_memory')
        
        # 从params读取Debug配置
        self.debug = getattr(params, 'debug', 0)
        self.vis_dir = None
    
    def initialize(self, image, info: dict):
        """初始化 - 生成初始记忆"""
        self.frame_id = 0
        self.state = list(info['init_bbox'])
        self.prev_image = image.copy()
        self.prev_bbox = list(info['init_bbox'])
        
        self.seq_name = info.get('seq_name', 'unknown')
        self.language_description = info.get('init_nlp', 'the target object')
        
        # 生成初始记忆
        self._generate_initial_memory(image, self.state)
        
        # 可视化目录
        if self.debug >= 2:
            env = env_settings()
            self.vis_dir = os.path.join(
                env.results_path, 'qwen_vlm_memory', 'vis', self.seq_name
            )
            os.makedirs(self.vis_dir, exist_ok=True)
        
        if self.debug >= 1:
            print(f"[QwenMemory] Initialize: {self.seq_name}")
            print(f"  Memory: {self.memory['appearance'][:60]}...")
    
    def _generate_initial_memory(self, image: np.ndarray, bbox: List[float]):
        """生成初始语义记忆"""
        img_with_box = draw_bbox(image, bbox, color=(0, 255, 0))
        prompt = get_prompt(self.init_prompt)
        
        try:
            output = self.vlm.infer([img_with_box], prompt)
            
            # 解析记忆
            output_clean = re.sub(r'```json\s*', '', output)
            output_clean = re.sub(r'```\s*', '', output_clean)
            data = json.loads(output_clean.strip())
            
            self.memory = {
                "appearance": dict_to_str(data.get("appearance", self.language_description)),
                "motion": dict_to_str(data.get("motion", "unknown")),
                "context": dict_to_str(data.get("context", "unknown")),
                "last_update": 0
            }
        except Exception as e:
            if self.debug >= 1:
                print(f"[QwenMemory] Memory init failed: {e}, using default")
            self.memory = {
                "appearance": self.language_description,
                "motion": "unknown",
                "context": "unknown",
                "last_update": 0
            }
    
    def track(self, image, info: dict = None):
        """记忆库跟踪"""
        self.frame_id += 1
        H, W = image.shape[:2]
        
        is_keyframe = info.get('is_keyframe', True) if info else True
        
        try:
            # 准备图像
            prev_with_box = draw_bbox(self.prev_image, self.prev_bbox, color=(0, 0, 255))
            
            # 构建prompt（包含记忆）
            prompt = get_prompt(
                self.track_prompt,
                memory_appearance=self.memory['appearance'],
                memory_motion=self.memory['motion'],
                memory_context=self.memory['context']
            )
            
            # VLM推理
            output = self.vlm.infer([prev_with_box, image], prompt)
            
            if self.debug >= 1:
                kf_tag = "[KF]" if is_keyframe else ""
                print(f"[QwenMemory] Frame {self.frame_id}{kf_tag}: {output[:80]}...")
            
            # 解析bbox
            bbox_xyxy = parse_bbox_from_text(output, W, H)
            
            if bbox_xyxy is not None:
                pred_bbox = xyxy_to_xywh(bbox_xyxy)
                self.state = pred_bbox
                
                # 尝试解析并更新记忆
                new_memory = parse_memory_state(output)
                if new_memory:
                    new_memory['last_update'] = self.frame_id
                    self.memory = new_memory
                    if self.debug >= 1:
                        print(f"  Memory updated")
                
                # 可视化
                if self.debug >= 2:
                    self._save_vis(prev_with_box, image, pred_bbox)
                
                # 更新上一帧
                self.prev_image = image.copy()
                self.prev_bbox = pred_bbox
            else:
                if self.debug >= 1:
                    print(f"[QwenMemory] Frame {self.frame_id}: Parse failed")
                self.state = [0, 0, 0, 0]
        
        except Exception as e:
            print(f"[QwenMemory] Error frame {self.frame_id}: {e}")
            import traceback
            traceback.print_exc()
            self.state = [0, 0, 0, 0]
        
        return {"target_bbox": self.state}
    
    def _save_vis(self, prev: np.ndarray, search: np.ndarray, bbox: List[float]):
        """保存可视化（带记忆信息）"""
        if not self.vis_dir:
            return
        
        result = draw_bbox(search, bbox, color=(255, 0, 0))
        combined = np.hstack([prev, result])
        
        # 添加记忆信息到padding
        h, w = combined.shape[:2]
        padding = np.ones((120, w, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        y = 25
        cv2.putText(padding, f"Frame {self.frame_id} - Memory:", (10, y), font, 0.6, (0, 0, 255), 2)
        y += 30
        cv2.putText(padding, f"App: {self.memory['appearance'][:90]}", (10, y), font, 0.4, (0, 0, 0), 1)
        y += 25
        cv2.putText(padding, f"Motion: {self.memory['motion'][:70]}", (10, y), font, 0.4, (0, 0, 0), 1)
        y += 25
        cv2.putText(padding, f"Context: {self.memory['context'][:70]}", (10, y), font, 0.4, (0, 0, 0), 1)
        
        final_img = np.vstack([combined, padding])
        
        vis_path = os.path.join(self.vis_dir, f"{self.seq_name}_{self.frame_id:04d}.jpg")
        cv2.imwrite(vis_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))


def get_tracker_class():
    return QwenVLMMemory

"""
Qwen VLM Tracker - 三图版本

三图跟踪：初始帧 + 上一帧 + 当前帧

特点:
- 固定初始帧作为目标锚点
- 上一帧提供运动连续性
- 减少逐帧漂移
"""
from lib.test.tracker.basetracker import BaseTracker
from lib.test.evaluation.environment import env_settings
import cv2
import numpy as np
import os
from typing import List

from lib.test.tracker.vlm_engine import VLMEngine
from lib.test.tracker.vlm_utils import parse_bbox_from_text, xyxy_to_xywh, draw_bbox
from lib.test.tracker.prompts import get_prompt


class QwenVLMThreeImage(BaseTracker):
    """
    三图VLM跟踪器
    
    跟踪范式:
    - Image 1: 初始帧 + 绿框 (固定锚点)
    - Image 2: 上一帧 + 蓝框 (运动线索)
    - Image 3: 当前帧
    → 输出: bbox预测
    """
    
    def __init__(self, params, dataset_name):
        super().__init__(params)
        self.params = params
        self.dataset_name = dataset_name
        
        # VLM引擎
        self.vlm = VLMEngine(params)
        
        # Tracker状态
        self.state = None
        self.init_image = None      # 固定: 初始帧
        self.init_bbox = None       # 固定: 初始框
        self.prev_image = None      # 动态: 上一帧
        self.prev_bbox = None       # 动态: 上一帧框
        self.language_description = None
        
        # 帧计数
        self.frame_id = 0
        self.seq_name = None
        
        # 从params读取Prompt配置
        self.prompt_name = getattr(params, 'prompt_name', 'three_image')
        
        # 从params读取Debug配置
        self.debug = getattr(params, 'debug', 0)
        self.vis_dir = None
    
    def initialize(self, image, info: dict):
        """初始化"""
        self.frame_id = 0
        self.state = list(info['init_bbox'])
        
        # 固定初始帧
        self.init_image = image.copy()
        self.init_bbox = list(info['init_bbox'])
        
        # 动态上一帧（初始化为初始帧）
        self.prev_image = image.copy()
        self.prev_bbox = list(info['init_bbox'])
        
        self.seq_name = info.get('seq_name', 'unknown')
        self.language_description = info.get('init_nlp', 'the target object')
        
        # 可视化目录
        if self.debug >= 2:
            env = env_settings()
            self.vis_dir = os.path.join(
                env.results_path, 'qwen_vlm_three', 'vis', self.seq_name
            )
            os.makedirs(self.vis_dir, exist_ok=True)
        
        if self.debug >= 1:
            print(f"[QwenThree] Initialize: {self.seq_name}, bbox={self.state}")
    
    def track(self, image, info: dict = None):
        """三图跟踪"""
        self.frame_id += 1
        H, W = image.shape[:2]
        
        is_keyframe = info.get('is_keyframe', True) if info else True
        
        try:
            # 准备三张图
            init_with_box = draw_bbox(self.init_image, self.init_bbox, color=(0, 255, 0))
            prev_with_box = draw_bbox(self.prev_image, self.prev_bbox, color=(0, 0, 255))
            current_img = image
            
            # 构建prompt
            prompt = get_prompt(
                self.prompt_name,
                target_description=self.language_description
            )
            
            # VLM三图推理
            output = self.vlm.infer([init_with_box, prev_with_box, current_img], prompt)
            
            if self.debug >= 1:
                kf_tag = "[KF]" if is_keyframe else ""
                print(f"[QwenThree] Frame {self.frame_id}{kf_tag}: {output[:80]}...")
            
            # 解析bbox
            bbox_xyxy = parse_bbox_from_text(output, W, H)
            
            if bbox_xyxy is not None:
                pred_bbox = xyxy_to_xywh(bbox_xyxy)
                self.state = pred_bbox
                
                # 可视化
                if self.debug >= 2:
                    self._save_vis(init_with_box, prev_with_box, image, pred_bbox)
                
                # 更新上一帧（短期记忆）
                self.prev_image = image.copy()
                self.prev_bbox = pred_bbox
            else:
                if self.debug >= 1:
                    print(f"[QwenThree] Frame {self.frame_id}: Parse failed")
                self.state = [0, 0, 0, 0]
        
        except Exception as e:
            print(f"[QwenThree] Error frame {self.frame_id}: {e}")
            self.state = [0, 0, 0, 0]
        
        return {"target_bbox": self.state}
    
    def _save_vis(self, init: np.ndarray, prev: np.ndarray, 
                  search: np.ndarray, bbox: List[float]):
        """保存三图可视化"""
        if not self.vis_dir:
            return
        
        result = draw_bbox(search, bbox, color=(255, 0, 0))
        combined = np.hstack([init, prev, result])
        
        vis_path = os.path.join(self.vis_dir, f"{self.seq_name}_{self.frame_id:04d}.jpg")
        cv2.imwrite(vis_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))


def get_tracker_class():
    return QwenVLMThreeImage

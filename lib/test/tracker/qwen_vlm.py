"""
Qwen VLM Tracker - 基础版本

最简单的VLM跟踪：模板帧 + 当前帧

特点:
- 两图跟踪（模板帧 + 当前帧）
- 动态更新模板
- 支持稀疏关键帧推理（由evaluation层控制）
"""
from lib.test.tracker.basetracker import BaseTracker
from lib.test.evaluation.environment import env_settings
import cv2
import numpy as np
import os
from typing import List

# 导入工具模块
from lib.test.tracker.vlm_engine import VLMEngine
from lib.test.tracker.vlm_utils import parse_bbox_from_text, xyxy_to_xywh, draw_bbox
from lib.test.tracker.prompts import get_prompt


class QwenVLMTracker(BaseTracker):
    """
    基础VLM跟踪器
    
    跟踪范式:
    - Image 1: 模板帧 + 绿框
    - Image 2: 当前帧
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
        self.template_image = None
        self.template_bbox = None
        self.language_description = None
        
        # 帧计数
        self.frame_id = 0
        self.seq_name = None
        
        # 从params读取Prompt配置
        self.prompt_name = getattr(params, 'prompt_name', 'two_image')
        
        # 从params读取Debug配置
        self.debug = getattr(params, 'debug', 0)
        self.vis_dir = None
    
    def initialize(self, image, info: dict):
        """初始化"""
        self.frame_id = 0
        self.state = list(info['init_bbox'])
        self.template_image = image.copy()
        self.template_bbox = list(info['init_bbox'])
        
        self.seq_name = info.get('seq_name', 'unknown')
        self.language_description = info.get('init_nlp', 'the target object')
        
        # 可视化目录
        if self.debug >= 2:
            env = env_settings()
            self.vis_dir = os.path.join(
                env.results_path, 'qwen_vlm', 'vis', self.seq_name
            )
            os.makedirs(self.vis_dir, exist_ok=True)
        
        if self.debug >= 1:
            print(f"[QwenVLM] Initialize: {self.seq_name}, bbox={self.state}")
    
    def track(self, image, info: dict = None):
        """跟踪"""
        self.frame_id += 1
        H, W = image.shape[:2]
        
        # info中的is_keyframe标记由evaluation层设置
        is_keyframe = info.get('is_keyframe', True) if info else True
        
        try:
            # 准备模板帧（带框）
            template_with_box = draw_bbox(
                self.template_image, self.template_bbox, color=(0, 255, 0)
            )
            
            # 构建prompt
            prompt = get_prompt(
                self.prompt_name,
                target_description=self.language_description
            )
            
            # VLM推理
            output = self.vlm.infer([template_with_box, image], prompt)
            
            if self.debug >= 1:
                kf_tag = "[KF]" if is_keyframe else ""
                print(f"[QwenVLM] Frame {self.frame_id}{kf_tag}: {output[:80]}...")
            
            # 解析bbox
            bbox_xyxy = parse_bbox_from_text(output, W, H)
            
            if bbox_xyxy is not None:
                pred_bbox = xyxy_to_xywh(bbox_xyxy)
                self.state = pred_bbox
                
                # 可视化
                if self.debug >= 2:
                    self._save_vis(template_with_box, image, pred_bbox)
                
                # 更新模板
                self.template_image = image.copy()
                self.template_bbox = pred_bbox
            else:
                if self.debug >= 1:
                    print(f"[QwenVLM] Frame {self.frame_id}: Parse failed")
                self.state = [0, 0, 0, 0]
        
        except Exception as e:
            print(f"[QwenVLM] Error frame {self.frame_id}: {e}")
            self.state = [0, 0, 0, 0]
        
        return {"target_bbox": self.state}
    
    def _save_vis(self, template: np.ndarray, search: np.ndarray, bbox: List[float]):
        """保存可视化"""
        if not self.vis_dir:
            return
        
        result = draw_bbox(search, bbox, color=(0, 0, 255))
        combined = np.hstack([template, result])
        
        vis_path = os.path.join(self.vis_dir, f"{self.frame_id:04d}.jpg")
        cv2.imwrite(vis_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))


def get_tracker_class():
    return QwenVLMTracker

"""
Qwen VLM Tracker - 认知跟踪（Cognitive Tracking）

通过语义记忆辅助跟踪，将语言认知能力引入目标跟踪：
    1. 初始化: 从初始帧生成语义记忆 {appearance, motion, context}
    2. 跟踪:   记忆 + 上一帧运动线索 → 预测bbox + 同步更新记忆
"""
from lib.test.tracker.basetracker import BaseTracker
from lib.test.evaluation.environment import env_settings
import cv2
import numpy as np
import os
from typing import List, Dict, Optional

from lib.test.tracker.vlm_engine import VLMEngine
from lib.test.tracker.vlm_utils import (
    parse_bbox_from_text, xyxy_to_xywh, draw_bbox,
    parse_memory_state
)
from lib.test.tracker.prompts import get_prompt

# 默认记忆结构
_EMPTY_MEMORY = {"appearance": "", "motion": "", "context": "", "last_update": 0}


class QwenVLMCognitive(BaseTracker):
    """
    认知VLM跟踪器

    跟踪范式:
        初始化: 用VLM分析初始帧，生成语义记忆
        跟踪:   上一帧+蓝框(运动线索) + 当前帧 + 语义记忆 → bbox + 记忆更新
    """

    def __init__(self, params, dataset_name):
        super().__init__(params)
        self.params = params
        self.dataset_name = dataset_name

        # VLM引擎
        self.vlm = VLMEngine(params)

        # 跟踪状态
        self.state: Optional[List[float]] = None
        self.prev_image: Optional[np.ndarray] = None
        self.prev_bbox: Optional[List[float]] = None
        self.language_description: str = "the target object"

        # 语义记忆库
        self.memory: Dict = dict(_EMPTY_MEMORY)

        # 配置
        self.track_prompt = getattr(params, 'track_prompt', 'memory_bank')
        self.init_prompt = getattr(params, 'init_prompt', 'init_memory')
        self.debug = getattr(params, 'debug', 0)

        self.frame_id = 0
        self.seq_name = None
        self.vis_dir = None

    def initialize(self, image, info: dict):
        """初始化 - 生成初始语义记忆"""
        self.frame_id = 0
        self.state = list(info['init_bbox'])
        self.prev_image = image.copy()
        self.prev_bbox = list(info['init_bbox'])

        self.seq_name = info.get('seq_name', 'unknown')
        lang = info.get('init_nlp', None)
        self.language_description = lang if lang else "the target object"

        # 生成初始记忆
        self._generate_initial_memory(image, self.state)

        if self.debug >= 2:
            env = env_settings()
            self.vis_dir = os.path.join(
                env.results_path, 'qwen_vlm_cognitive', 'vis', self.seq_name
            )
            os.makedirs(self.vis_dir, exist_ok=True)

        if self.debug >= 1:
            print(f"[Cognitive] Init '{self.seq_name}' | "
                  f"memory: {self.memory['appearance'][:50]}...")

    def _generate_initial_memory(self, image: np.ndarray, bbox: List[float]):
        """调用VLM生成初始语义记忆，使用parse_memory_state统一解析"""
        img_with_box = draw_bbox(image, bbox, color=(0, 255, 0))
        prompt = get_prompt(self.init_prompt)

        try:
            output = self.vlm.infer([img_with_box], prompt)
            parsed = parse_memory_state(output)
            if parsed:
                self.memory = {**parsed, "last_update": 0}
            else:
                raise ValueError("parse_memory_state returned None")
        except Exception as e:
            if self.debug >= 1:
                print(f"[Cognitive] Memory init failed ({e}), using description fallback")
            self.memory = {
                "appearance": self.language_description,
                "motion":     "unknown",
                "context":    "unknown",
                "last_update": 0
            }

    def track(self, image, info: dict = None):
        """认知跟踪: 记忆 + 运动线索 → bbox + 记忆更新"""
        self.frame_id += 1
        H, W = image.shape[:2]

        try:
            prev_with_box = draw_bbox(self.prev_image, self.prev_bbox, color=(0, 0, 255))

            # Prompt中注入当前记忆
            prompt = get_prompt(
                self.track_prompt,
                memory_appearance=self.memory['appearance'],
                memory_motion=self.memory['motion'],
                memory_context=self.memory['context']
            )

            output = self.vlm.infer([prev_with_box, image], prompt)

            if self.debug >= 1:
                kf_tag = "[KF]" if (info and info.get('is_keyframe')) else ""
                print(f"[Cognitive] Frame {self.frame_id}{kf_tag}: {output[:100]}")

            bbox_xyxy = parse_bbox_from_text(output, W, H)

            if bbox_xyxy is not None:
                pred_bbox = xyxy_to_xywh(bbox_xyxy)
                self.state = pred_bbox

                # 尝试更新记忆
                new_memory = parse_memory_state(output)
                if new_memory:
                    new_memory['last_update'] = self.frame_id
                    self.memory = new_memory

                # 更新上一帧
                self.prev_image = image.copy()
                self.prev_bbox = pred_bbox

                if self.debug >= 2:
                    self._save_vis(prev_with_box, image, pred_bbox)
            else:
                # Parse失败: last-frame fallback
                if self.debug >= 1:
                    print(f"[Cognitive] Frame {self.frame_id}: parse failed, keeping last state")

        except Exception as e:
            if self.debug >= 1:
                print(f"[Cognitive] Frame {self.frame_id} error: {e}")
            import traceback
            traceback.print_exc()

        return {"target_bbox": self.state}

    def _save_vis(self, prev: np.ndarray, current: np.ndarray, bbox: List[float]):
        """保存带记忆信息的可视化"""
        if not self.vis_dir:
            return
        result = draw_bbox(current, bbox, color=(255, 0, 0))
        combined = np.hstack([prev, result])
        h, w = combined.shape[:2]
        pad = np.ones((100, w, 3), dtype=np.uint8) * 255
        font, sc = cv2.FONT_HERSHEY_SIMPLEX, 0.4
        cv2.putText(pad, f"F{self.frame_id} App: {self.memory['appearance'][:80]}", (8, 24), font, sc, (0,0,0), 1)
        cv2.putText(pad, f"Motion: {self.memory['motion'][:80]}",  (8, 50), font, sc, (0,0,0), 1)
        cv2.putText(pad, f"Context: {self.memory['context'][:80]}", (8, 76), font, sc, (0,0,0), 1)
        vis_path = os.path.join(self.vis_dir, f"{self.frame_id:04d}.jpg")
        cv2.imwrite(vis_path, cv2.cvtColor(np.vstack([combined, pad]), cv2.COLOR_RGB2BGR))


def get_tracker_class():
    return QwenVLMCognitive

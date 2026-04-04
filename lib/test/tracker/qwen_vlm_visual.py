"""
Qwen VLM Tracker - 纯视觉跟踪（Visual Tracking）

通过 num_frames 参数统一控制两图/三图两种范式：
    num_frames=2: [初始帧+绿框, 当前帧]          ← 简洁，适合短时跟踪
    num_frames=3: [初始帧+绿框, 上一帧+蓝框, 当前帧]  ← 含运动线索，减少漂移
"""
from lib.test.tracker.basetracker import BaseTracker
from lib.test.evaluation.environment import env_settings
import cv2
import numpy as np
import os
from typing import List, Optional

from lib.test.tracker.vlm_engine import VLMEngine
from lib.test.tracker.vlm_utils import parse_bbox_from_text, xyxy_to_xywh, draw_bbox
from lib.test.tracker.prompts import get_prompt


class QwenVLMVisual(BaseTracker):
    """
    纯视觉VLM跟踪器

    跟踪范式:
        num_frames=2 (two-image):
            Image 1: 初始帧 + 绿框 (固定锚点)
            Image 2: 当前帧
        num_frames=3 (three-image):
            Image 1: 初始帧 + 绿框 (固定锚点)
            Image 2: 上一帧 + 蓝框 (运动线索)
            Image 3: 当前帧
    """

    def __init__(self, params, dataset_name):
        super().__init__(params)
        self.params = params
        self.dataset_name = dataset_name

        # VLM引擎
        self.vlm = VLMEngine(params)

        # 跟踪状态
        self.state: Optional[List[float]] = None
        self.init_image: Optional[np.ndarray] = None
        self.init_bbox: Optional[List[float]] = None
        self.prev_image: Optional[np.ndarray] = None   # 仅 num_frames=3 使用
        self.prev_bbox: Optional[List[float]] = None
        self.language_description: str = "the target object"

        # 配置
        self.num_frames = getattr(params, 'num_frames', 2)
        self.prompt_name = getattr(params, 'prompt_name', 'two_image' if self.num_frames == 2 else 'three_image')
        self.debug = getattr(params, 'debug', 0)

        self.frame_id = 0
        self.seq_name = None
        self.vis_dir = None
        self.init_frame_num = 0
        self.prev_frame_num = 0

    def initialize(self, image, info: dict):
        """初始化，固定初始帧"""
        self.frame_id = 0
        self.state = list(info['init_bbox'])
        self.init_image = image.copy()
        self.init_bbox = list(info['init_bbox'])
        self.prev_image = image.copy()
        self.prev_bbox = list(info['init_bbox'])
        self.init_frame_num = info.get('frame_num', 0)
        self.prev_frame_num = info.get('frame_num', 0)

        self.seq_name = info.get('seq_name', 'unknown')
        self.run_tag = info.get('run_tag', None)
        lang = info.get('init_nlp', None)
        self.language_description = lang if lang else "the target object"

        if self.debug >= 2:
            env = env_settings()
            vis_root = os.path.join(env.results_path, 'qwen_vlm_visual', 'vis')
            if self.run_tag:
                self.vis_dir = os.path.join(vis_root, self.run_tag, self.seq_name)
            else:
                self.vis_dir = os.path.join(vis_root, self.seq_name)
            os.makedirs(self.vis_dir, exist_ok=True)

        if self.debug >= 1:
            print(f"[Visual] Init '{self.seq_name}' num_frames={self.num_frames}")

    def track(self, image, info: dict = None):
        """纯视觉跟踪"""
        self.frame_id += 1
        H, W = image.shape[:2]
        current_frame_num = info.get('frame_num', self.frame_id) if info else self.frame_id
        motion_frame_num = self.prev_frame_num

        try:
            # 构建输入图像序列
            init_with_box = draw_bbox(self.init_image, self.init_bbox, color=(0, 255, 0))

            if self.num_frames == 3:
                prev_with_box = draw_bbox(self.prev_image, self.prev_bbox, color=(0, 0, 255))
                images = [init_with_box, prev_with_box, image]
            else:
                images = [init_with_box, image]

            # 构建Prompt
            prompt = get_prompt(self.prompt_name, target_description=self.language_description)

            # VLM推理
            output = self.vlm.infer(images, prompt)

            if self.debug >= 1:
                kf_tag = "[KF]" if (info and info.get('is_keyframe')) else ""
                print(f"[Visual] Frame {self.frame_id}{kf_tag}: {output[:100]}")

            # 解析bbox
            bbox_xyxy = parse_bbox_from_text(output, W, H)

            if bbox_xyxy is not None:
                pred_bbox = xyxy_to_xywh(bbox_xyxy)
                self.state = pred_bbox
                # 更新上一帧（仅 num_frames=3 使用）
                self.prev_image = image.copy()
                self.prev_bbox = pred_bbox
                self.prev_frame_num = current_frame_num

                if self.debug >= 2:
                    panel_titles = [f"Template Frame: {self.init_frame_num}"]
                    if self.num_frames == 3:
                        panel_titles.append(f"Motion Frame: {motion_frame_num}")
                    panel_titles.append(f"Search Frame: {current_frame_num}")
                    self._save_vis(images, pred_bbox, panel_titles, current_frame_num)
            else:
                # Parse失败: last-frame fallback，保持 self.state 不变
                if self.debug >= 1:
                    print(f"[Visual] Frame {self.frame_id}: parse failed, keeping last state")

        except Exception as e:
            if self.debug >= 1:
                print(f"[Visual] Frame {self.frame_id} error: {e}")

        return {"target_bbox": self.state}

    def _annotate_panel(self, image: np.ndarray, title: str) -> np.ndarray:
        annotated = image.copy()
        cv2.rectangle(annotated, (0, 0), (320, 30), (255, 255, 255), -1)
        cv2.putText(annotated, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        return annotated

    def _save_vis(self, images: List[np.ndarray], bbox: List[float],
                  panel_titles: List[str], current_frame_num: int):
        """保存可视化结果"""
        if not self.vis_dir:
            return
        result = draw_bbox(images[-1], bbox, color=(255, 0, 0))
        panels = [*images[:-1], result]
        annotated_panels = [
            self._annotate_panel(panel, title)
            for panel, title in zip(panels, panel_titles)
        ]
        combined = np.hstack(annotated_panels)
        vis_path = os.path.join(self.vis_dir, f"{current_frame_num:06d}.jpg")
        cv2.imwrite(vis_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))


def get_tracker_class():
    return QwenVLMVisual

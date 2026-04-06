"""
Qwen VLM Tracker - 认知跟踪（Cognitive Tracking）

通过语义记忆辅助跟踪，将语言认知能力引入目标跟踪：
    1. 初始化: 从初始帧生成语义记忆 {appearance, motion, context}
    2. 跟踪:   记忆 + 全图搜索 → 结构化状态判断 + bbox + 认知推理文本

核心改进（2026-03-17）：
    - 全图搜索：不依赖上一帧位置先验
    - 结构化输出：target_status（6种状态）+ environment_status（9种环境）
    - 认知推理：tracking_evidence（主观性文本描述）
"""
from lib.test.tracker.basetracker import BaseTracker
from lib.test.evaluation.environment import env_settings
import cv2
import numpy as np
import os
from typing import List, Dict, Optional

from lib.test.tracker.vlm_engine import VLMEngine
from lib.test.tracker.vlm_utils import (
    parse_cognitive_output, xyxy_to_xywh, draw_bbox,
    parse_memory_state
)
from lib.test.tracker.prompts import get_prompt

# 默认记忆结构
_EMPTY_MEMORY = {"appearance": "", "motion": "", "context": "", "last_update": 0}


class VLMCognitive(BaseTracker):
    """
    认知VLM跟踪器（新版）

    跟踪范式:
        初始化: 用VLM分析初始帧，生成语义记忆
        跟踪:   上一帧+蓝框(仅用于运动估计) + 当前帧 + 语义记忆 → 结构化状态判断 + bbox + 认知推理

    核心改进:
        - 全图搜索：Prompt 明确要求忽略位置先验
        - 结构化输出：target_status + environment_status + tracking_evidence
        - 认知推理：输出主观性文本描述
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

        # 认知跟踪状态（新增）
        self.target_status: str = "normal"
        self.environment_status: List[str] = ["normal"]
        self.tracking_evidence: str = ""
        self.tracking_history: List[Dict] = []  # 保存跟踪历史用于分析

        # 配置
        self.track_prompt = getattr(params, 'track_prompt', 'cognitive')
        self.init_prompt = getattr(params, 'init_prompt', 'init_memory')
        self.use_soi_text = getattr(params, 'use_soi_text', False)  # 是否使用 SOI 文本描述
        self.debug = getattr(params, 'debug', 0)

        self.frame_id = 0
        self.seq_name = None
        self.vis_dir = None
        self.prev_frame_num = 0

    def initialize(self, image, info: dict):
        """初始化 - 生成初始语义记忆"""
        self.frame_id = 0
        self.state = list(info['init_bbox'])
        self.prev_image = image.copy()
        self.prev_bbox = list(info['init_bbox'])
        self.prev_frame_num = info.get('frame_num', 0)

        self.seq_name = info.get('seq_name', 'unknown')
        self.run_tag = info.get('run_tag', None)
        lang = info.get('init_nlp', None)
        self.language_description = lang if lang else "the target object"

        # 生成初始记忆
        self._generate_initial_memory(image, self.state)

        if self.debug >= 2:
            env = env_settings()
            vis_root = os.path.join(env.results_path, 'vlm_cognitive', 'vis')
            if self.run_tag:
                self.vis_dir = os.path.join(vis_root, self.run_tag, self.seq_name)
            else:
                self.vis_dir = os.path.join(vis_root, self.seq_name)
            os.makedirs(self.vis_dir, exist_ok=True)

        if self.debug >= 1:
            print(f"[Cognitive] Init '{self.seq_name}' | "
                  f"memory: {self.memory['appearance'][:50]}...")

    def _generate_initial_memory(self, image: np.ndarray, bbox: List[float]):
        """调用VLM生成初始语义记忆，使用parse_memory_state统一解析"""
        img_with_box = draw_bbox(image, bbox, color=(0, 255, 0))
        prompt = get_prompt(self.init_prompt, target_description=self.language_description)

        try:
            output = self.vlm.infer([img_with_box], prompt)
            parsed = parse_memory_state(output)
            if parsed:
                desc = (self.language_description or "").strip()
                if desc and desc.lower() not in ("the target object", "target object"):
                    parsed['appearance'] = f"{desc}; {parsed.get('appearance', '')}".strip("; ")
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
        """认知跟踪: 记忆 + 全图搜索 → 结构化状态判断 + bbox + 认知推理"""
        self.frame_id += 1
        H, W = image.shape[:2]
        current_frame_num = info.get('frame_num', self.frame_id) if info else self.frame_id
        template_frame_num = self.prev_frame_num

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
                print(f"[Cognitive] Frame {self.frame_id}{kf_tag}")
                print(f"  Raw output: {output[:150]}...")

            # 解析结构化输出
            result = parse_cognitive_output(output, W, H)

            if result:
                if self.debug >= 1:
                    print(f"  Status: {result['target_status']}")
                    print(f"  Environment: {result['environment_status']}")
                    print(f"  Confidence: {result['confidence']:.2f}")
                    print(f"  Evidence: {result['tracking_evidence'][:80]}...")

                # 更新状态
                self.state = result['bbox']
                self.target_status = result['target_status']
                self.environment_status = result['environment_status']
                self.tracking_evidence = result['tracking_evidence']

                # 只在目标可见时更新 prev_image/prev_bbox
                visible_status = ['normal', 'partially_occluded', 'reappeared']
                if result['target_status'] in visible_status and result['bbox'] != [0, 0, 0, 0]:
                    self.prev_image = image.copy()
                    self.prev_bbox = result['bbox']
                    self.prev_frame_num = current_frame_num

                # 保存跟踪历史
                self.tracking_history.append({
                    'frame_id': self.frame_id,
                    'target_status': result['target_status'],
                    'environment_status': result['environment_status'],
                    'bbox': result['bbox'],
                    'confidence': result['confidence'],
                    'tracking_evidence': result['tracking_evidence']
                })

                # 用当前帧的认知输出更新语义记忆
                self.memory = {
                    "appearance": self.memory['appearance'],  # 保持外观记忆稳定
                    "motion": result['tracking_evidence'][:100],  # 用 evidence 更新运动/状态
                    "context": self.memory['context'],
                    "last_update": self.frame_id
                }

                if self.debug >= 2:
                    self._save_vis(prev_with_box, image, result, template_frame_num, current_frame_num)

            else:
                # Parse失败: 标记为不可见
                if self.debug >= 1:
                    print(f"[Cognitive] Frame {self.frame_id}: Parse failed, marking as absent")
                self.state = [0, 0, 0, 0]
                self.target_status = "disappeared"
                self.environment_status = ["normal"]

        except Exception as e:
            if self.debug >= 1:
                print(f"[Cognitive] Frame {self.frame_id} error: {e}")
            import traceback
            traceback.print_exc()
            self.state = [0, 0, 0, 0]

        return {"target_bbox": self.state}

    def _annotate_panel(self, image: np.ndarray, title: str) -> np.ndarray:
        annotated = image.copy()
        cv2.rectangle(annotated, (0, 0), (320, 30), (255, 255, 255), -1)
        cv2.putText(annotated, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        return annotated

    def _save_vis(self, prev: np.ndarray, current: np.ndarray, result: dict,
                  template_frame_num: int, search_frame_num: int):
        """保存带认知信息的可视化"""
        if not self.vis_dir:
            return

        bbox = result['bbox']
        if bbox != [0, 0, 0, 0]:
            result_img = draw_bbox(current, bbox, color=(255, 0, 0))
        else:
            result_img = current.copy()

        prev_annotated = self._annotate_panel(prev, f"Template Frame: {template_frame_num}")
        result_annotated = self._annotate_panel(result_img, f"Search Frame: {search_frame_num}")
        combined = np.hstack([prev_annotated, result_annotated])
        _, w = combined.shape[:2]
        pad = np.ones((170, w, 3), dtype=np.uint8) * 255
        font, sc = cv2.FONT_HERSHEY_SIMPLEX, 0.4

        cv2.putText(pad, f"Seq Frame: {search_frame_num} | Tracker Step: {self.frame_id}", (8, 24), font, sc, (0, 0, 0), 1)
        cv2.putText(pad, f"Status: {result['target_status']}", (8, 50), font, sc, (0, 0, 0), 1)
        # cv2.putText(pad, f"Template Frame: {template_frame_num}", (260, 50), font, sc, (0, 0, 0), 1)
        # cv2.putText(pad, f"Search Frame: {search_frame_num}", (520, 50), font, sc, (0, 0, 0), 1)
        cv2.putText(pad, f"Environment: {', '.join(result['environment_status'])}", (8, 76), font, sc, (0, 0, 0), 1)
        cv2.putText(pad, f"Confidence: {result['confidence']:.2f}", (8, 102), font, sc, (0, 0, 0), 1)

        evidence = result['tracking_evidence'][:120]
        cv2.putText(pad, f"Evidence: {evidence}", (8, 128), font, sc, (0, 0, 255), 1)
        cv2.putText(pad, f"Memory: {self.memory['appearance'][:80]}", (8, 154), font, sc, (0, 100, 0), 1)

        vis_path = os.path.join(self.vis_dir, f"{search_frame_num:06d}.jpg")
        cv2.imwrite(vis_path, cv2.cvtColor(np.vstack([combined, pad]), cv2.COLOR_RGB2BGR))


def get_tracker_class():
    return VLMCognitive

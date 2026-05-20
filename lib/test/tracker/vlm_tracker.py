"""
Unified VLM tracker for CognitiveBench.

This tracker keeps the classic tracking interface while using a fixed MLLM
output schema:
    target_status: present | absent
    bbox: [x1, y1, x2, y2] in model output, converted to xywh pixels
    reasoning: str
    target_text: str
"""
import json
import math
import os
import textwrap
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.prompts import get_prompt
from lib.test.tracker.vlm_engine import VLMEngine
from lib.test.tracker.vlm_utils import (
    convert_to_pixel_bbox,
    draw_bbox,
    fix_and_clip_bbox,
    strip_code_fence,
    xyxy_to_xywh,
)


_NAN_BBOX = [float('nan')] * 4


class VLMTracker(BaseTracker):
    """Single configurable MLLM tracker entrypoint."""

    def __init__(self, params, dataset_name):
        super().__init__(params)
        self.params = params
        self.dataset_name = dataset_name
        self.vlm = VLMEngine(params)

        self.context_mode = getattr(params, 'context_mode', 'mosaic')
        self.history_policy = getattr(params, 'history_policy', 'visible_keyframes')
        self.history_buffer_size = int(getattr(params, 'history_buffer_size', 3))
        self.use_init_anchor = bool(getattr(params, 'use_init_anchor', True))
        self.use_init_bbox_ref = bool(getattr(params, 'use_init_bbox_ref', False))
        self.prompt_name = getattr(params, 'prompt_name', 'cognitivebench')
        self.vlm_max_image_side = int(getattr(params, 'vlm_max_image_side', 648))
        self.output_reasoning = bool(getattr(params, 'output_reasoning', True))
        self.target_text_history_size = int(getattr(params, 'target_text_history_size', 3))
        self.debug = int(getattr(params, 'debug', 0))
        self.vis_dir = None

        self.frame_id = 0
        self.seq_name = None
        self.language_description = ""
        self.target_text_history: List[str] = []

        self.state: List[float] = list(_NAN_BBOX)
        self.init_image: Optional[np.ndarray] = None
        self.init_bbox: Optional[List[float]] = None
        self.history_buffer: List[Tuple[int, np.ndarray, List[float], str]] = []

    def initialize(self, image, info: dict):
        self.frame_id = int(info.get('frame_num', 0))
        self.seq_name = info.get('seq_name', 'unknown')
        self.language_description = info.get('init_nlp', '') or ''
        self.target_text_history = []
        self.init_image = image.copy()
        self.init_bbox = list(info['init_bbox'])
        self.state = list(info['init_bbox'])
        self.history_buffer = []

        if self.debug >= 2:
            results_dir = getattr(self.params, 'results_dir', None)
            if results_dir:
                dataset_dir = info.get('run_tag', self.dataset_name)
                self.vis_dir = os.path.join(results_dir, 'vis', str(dataset_dir), self.seq_name)
            else:
                self.vis_dir = os.path.join('debug_vis', 'vlm_tracker', self.seq_name)
            os.makedirs(self.vis_dir, exist_ok=True)

        if self.debug >= 1:
            print(f"[VLMTracker] Init {self.seq_name} context={self.context_mode}")

        return {
            "target_bbox": self.state,
            "vlm_metadata": {
                "target_status": "present",
                "bbox_xyxy": self._xywh_to_xyxy(self.state),
                "reasoning": "Initial ground-truth target annotation.",
                "target_text": self.language_description,
                "target_text_history": list(self.target_text_history),
                "context_mode": self.context_mode,
            }
        }

    def track(self, image, info: dict = None):
        if info is not None and 'frame_num' in info:
            self.frame_id = int(info['frame_num'])
        else:
            self.frame_id += 1

        H, W = image.shape[:2]

        try:
            images, prompt_kwargs, image_meta = self._build_context(image, W, H)
            prompt = get_prompt(self.prompt_name, **prompt_kwargs)
            raw_output = self.vlm.infer(images, prompt)
            result = parse_cognitivebench_output(raw_output, W, H)

            if result is None:
                result = {
                    "target_status": "absent",
                    "bbox_xyxy": None,
                    "target_bbox": list(_NAN_BBOX),
                    "reasoning": "Failed to parse model output.",
                    "target_text": "",
                }

            self.state = result["target_bbox"]
            self._update_history(image, result, bool(info and info.get('is_keyframe')))
            self._update_target_text(result)

            metadata = {
                "target_status": result["target_status"],
                "bbox_xyxy": result["bbox_xyxy"] if result["bbox_xyxy"] is not None else list(_NAN_BBOX),
                "reasoning": result.get("reasoning", ""),
                "target_text": result.get("target_text", ""),
                "target_text_history": list(self.target_text_history),
                "raw_output": raw_output,
                "context_mode": self.context_mode,
                "history_size": len(self.history_buffer),
                "vlm_input_sizes": image_meta["vlm_input_sizes"],
                "original_current_size": image_meta["original_current_size"],
            }

            if self.debug >= 1:
                print(f"[VLMTracker] F{self.frame_id} {metadata['target_status']} "
                      f"bbox={self.state}")

            if self.debug >= 2:
                self._save_vis(images, result, metadata)

            return {
                "target_bbox": self.state,
                "vlm_metadata": metadata,
            }

        except Exception as e:
            if self.debug >= 1:
                print(f"[VLMTracker] F{self.frame_id} error: {e}")
            self.state = list(_NAN_BBOX)
            return {
                "target_bbox": self.state,
                "vlm_metadata": {
                    "target_status": "absent",
                    "bbox_xyxy": list(_NAN_BBOX),
                    "reasoning": f"Tracker exception: {e}",
                    "target_text": "",
                    "target_text_history": list(self.target_text_history),
                    "raw_output": "",
                    "context_mode": self.context_mode,
                    "history_size": len(self.history_buffer),
                }
            }

    def _build_context(self, image: np.ndarray, W: int, H: int):
        init_bbox_1000 = self._init_bbox_1000() if self.use_init_bbox_ref else None
        prompt_kwargs = {
            "context_mode": self.context_mode,
            "language_description": self.language_description,
            "target_text_history": list(self.target_text_history),
            "output_reasoning": self.output_reasoning,
            "num_history_frames": len(self.history_buffer),
            "use_init_bbox_ref": self.use_init_bbox_ref,
            "init_bbox_1000": init_bbox_1000,
        }

        if self.context_mode == 'pair':
            init_ref = draw_bbox(self.init_image, self.init_bbox, color=(0, 255, 0))
            images = [init_ref, image]
        else:
            mosaic = self._create_mosaic()
            images = [mosaic, image]

        resized_images, input_sizes = self._resize_images_for_vlm(images)
        image_meta = {
            "vlm_input_sizes": input_sizes,
            "original_current_size": [W, H],
        }
        return resized_images, prompt_kwargs, image_meta

    def _create_mosaic(self, target_height: int = 240) -> np.ndarray:
        frames = []
        if self.use_init_anchor:
            frames.append((0, self.init_image, self.init_bbox, "init", True))
        for fid, img, bbox, status in self.history_buffer:
            frames.append((fid, img, bbox, status, False))

        if not frames:
            return self.init_image.copy()

        panels = []
        for idx, (fid, img, bbox, status, is_gt) in enumerate(frames):
            panel = img.copy()
            if bbox is not None and self._bbox_valid(bbox):
                color = (0, 255, 0) if is_gt else (255, 0, 0)
                panel = draw_bbox(panel, bbox, color=color, thickness=2)

            h, w = panel.shape[:2]
            scale = target_height / h
            new_w = max(1, int(w * scale))
            panel = cv2.resize(panel, (new_w, target_height))

            header = np.ones((30, new_w, 3), dtype=np.uint8) * 255
            label = f"#{fid} {'GT' if is_gt else status}"
            cv2.putText(header, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            panels.append(np.vstack([header, panel]))

            if idx < len(frames) - 1:
                panels.append(np.ones((target_height + 30, 5, 3), dtype=np.uint8) * 220)

        return np.hstack(panels)

    def _resize_images_for_vlm(self, images: List[np.ndarray]):
        resized = []
        sizes = []
        for image in images:
            out = self._resize_long_side(image, self.vlm_max_image_side)
            h, w = out.shape[:2]
            resized.append(out)
            sizes.append([w, h])
        return resized, sizes

    def _save_vis(self, vlm_images: List[np.ndarray], result: Dict, metadata: Dict):
        if not self.vis_dir or len(vlm_images) < 2:
            return

        reference_img = vlm_images[0]
        current_img = vlm_images[1].copy()
        pred_bbox = result.get("target_bbox")

        if result.get("target_status") == "present" and self._bbox_valid(pred_bbox):
            orig_w, orig_h = metadata.get("original_current_size", [current_img.shape[1], current_img.shape[0]])
            cur_h, cur_w = current_img.shape[:2]
            scale_x = cur_w / float(orig_w)
            scale_y = cur_h / float(orig_h)
            scaled_bbox = [
                pred_bbox[0] * scale_x,
                pred_bbox[1] * scale_y,
                pred_bbox[2] * scale_x,
                pred_bbox[3] * scale_y,
            ]
            current_img = draw_bbox(current_img, scaled_bbox, color=(255, 0, 0), thickness=2)

        top = self._hstack_with_common_height(reference_img, current_img, target_height=360)
        panel = self._make_text_panel(top.shape[1], metadata)
        vis = np.vstack([top, panel])

        save_path = os.path.join(self.vis_dir, f"{self.frame_id:06d}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    @staticmethod
    def _hstack_with_common_height(left: np.ndarray, right: np.ndarray, target_height: int = 360) -> np.ndarray:
        def resize_to_height(img):
            h, w = img.shape[:2]
            if h == target_height:
                return img
            new_w = max(1, int(round(w * target_height / float(h))))
            return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)

        left = resize_to_height(left)
        right = resize_to_height(right)
        sep = np.ones((target_height, 6, 3), dtype=np.uint8) * 230
        return np.hstack([left, sep, right])

    def _make_text_panel(self, width: int, metadata: Dict) -> np.ndarray:
        panel_h = 170
        panel = np.ones((panel_h, width, 3), dtype=np.uint8) * 245
        font = cv2.FONT_HERSHEY_SIMPLEX
        x, y = 10, 24

        lines = [
            f"Frame {self.frame_id} | status={metadata.get('target_status', '')} | context={metadata.get('context_mode', '')}",
            f"bbox_xyxy={metadata.get('bbox_xyxy', '')}",
            f"target_text={metadata.get('target_text', '')}",
            f"text_history={' | '.join(metadata.get('target_text_history', []))}",
        ]
        reasoning = metadata.get("reasoning", "")
        if reasoning:
            lines.append(f"reasoning={reasoning}")
        raw_output = str(metadata.get("raw_output", ""))
        if raw_output:
            lines.append(f"raw={raw_output[:220]}")

        max_chars = max(40, width // 9)
        draw_lines = []
        for line in lines:
            draw_lines.extend(textwrap.wrap(str(line), width=max_chars) or [""])

        for line in draw_lines[:7]:
            cv2.putText(panel, line, (x, y), font, 0.45, (25, 25, 25), 1, cv2.LINE_AA)
            y += 22
        return panel

    @staticmethod
    def _resize_long_side(image: np.ndarray, max_side: int) -> np.ndarray:
        if max_side <= 0:
            return image

        h, w = image.shape[:2]
        long_side = max(h, w)
        if long_side <= max_side:
            return image

        scale = max_side / float(long_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _update_history(self, image: np.ndarray, result: Dict, is_keyframe: bool):
        if self.history_policy == 'none':
            return
        if self.history_policy == 'visible_keyframes' and not is_keyframe:
            return

        visible = result.get("target_status") == "present" and self._bbox_valid(result.get("target_bbox"))
        if self.history_policy in {'visible_keyframes', 'visible_all'} and not visible:
            return
        if self.history_policy == 'all_keyframes' and not is_keyframe:
            return

        bbox = result["target_bbox"] if visible else list(_NAN_BBOX)
        self.history_buffer.append((self.frame_id, image.copy(), bbox, result.get("target_status", "absent")))
        if len(self.history_buffer) > self.history_buffer_size:
            self.history_buffer.pop(0)

    def _update_target_text(self, result: Dict):
        if result.get("target_status") != "present":
            return

        target_text = str(result.get("target_text", "")).strip()
        if not target_text:
            return

        if self.target_text_history and self.target_text_history[-1] == target_text:
            return

        self.target_text_history.append(target_text)
        if len(self.target_text_history) > self.target_text_history_size:
            self.target_text_history.pop(0)

    def _init_bbox_1000(self):
        if self.init_image is None or self.init_bbox is None:
            return None
        h, w = self.init_image.shape[:2]
        x, y, bw, bh = self.init_bbox
        return [
            int(max(0, min(999, x / w * 1000))),
            int(max(0, min(999, y / h * 1000))),
            int(max(0, min(999, (x + bw) / w * 1000))),
            int(max(0, min(999, (y + bh) / h * 1000))),
        ]

    @staticmethod
    def _bbox_valid(bbox):
        if bbox is None or len(bbox) != 4:
            return False
        return all(isinstance(v, (int, float)) and math.isfinite(v) for v in bbox) and bbox[2] > 0 and bbox[3] > 0

    @staticmethod
    def _xywh_to_xyxy(bbox):
        x, y, w, h = bbox
        return [x, y, x + w, y + h]


def parse_cognitivebench_output(text: str, img_width: int, img_height: int) -> Optional[Dict]:
    try:
        data = json.loads(strip_code_fence(text))
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    status = str(data.get("target_status", "")).strip().lower()
    if status not in {"present", "absent"}:
        return None

    reasoning = data.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)

    target_text = data.get("target_text", "")
    if not isinstance(target_text, str):
        target_text = str(target_text)
    target_text = target_text.strip()

    if status == "absent":
        return {
            "target_status": "absent",
            "bbox_xyxy": None,
            "target_bbox": list(_NAN_BBOX),
            "reasoning": reasoning,
            "target_text": target_text,
        }

    bbox = data.get("bbox", None)
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    if any(v is None for v in bbox):
        return None

    try:
        bbox = [float(v) for v in bbox]
    except Exception:
        return None

    bbox_xyxy = convert_to_pixel_bbox(bbox, img_width, img_height)
    bbox_xyxy = fix_and_clip_bbox(bbox_xyxy, img_width, img_height)
    bbox_xywh = xyxy_to_xywh(bbox_xyxy)

    return {
        "target_status": "present",
        "bbox_xyxy": bbox_xyxy,
        "target_bbox": bbox_xywh,
        "reasoning": reasoning,
        "target_text": target_text,
    }


def get_tracker_class():
    return VLMTracker

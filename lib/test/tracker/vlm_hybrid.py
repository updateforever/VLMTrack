"""
Qwen VLM Tracker - 混合跟踪（Hybrid Tracking）

结合传统 SOT 算法（SUTrack）的高速稳定性与 VLM 的语义理解能力：

设计思路:
    - SUTrack 负责所有帧的基础跟踪（高速、连续）
    - VLM 按触发策略介入做语义校正
    - VLM 成功校正后立即重置 SUTrack 状态，防止漂移累积

VLM 模式 (vlm_mode):
    'visual'   - 纯视觉范式（初始帧+[上一帧]+当前帧）
    'cognitive' - 认知范式（语义记忆+上一帧+当前帧）

触发策略 (trigger_mode):
    'keyframe'   - 外部关键帧索引控制（确定性）
    'confidence' - SUTrack 置信度低于阈值时自动触发
    'hybrid'     - 两者满足其一即触发
"""
import torch
import numpy as np
import cv2
import os
from typing import List, Optional, Dict

from lib.test.tracker.basetracker import BaseTracker
from lib.test.evaluation.environment import env_settings

# VLM相关
from lib.test.tracker.vlm_engine import VLMEngine
from lib.test.tracker.vlm_utils import (
    parse_bbox_from_text, xyxy_to_xywh, draw_bbox,
    parse_memory_state
)
from lib.test.tracker.prompts import get_prompt

# SUTrack相关
from lib.models.sutrack import build_sutrack
from lib.test.tracker.utils import sample_target, transform_image_to_crop, Preprocessor
from lib.utils.box_ops import clip_box
from lib.test.utils.hann import hann2d


class VLMHybrid(BaseTracker):
    """
    混合VLM跟踪器

    SUTrack 提供连续、高速的基础跟踪；
    VLM 在关键帧或置信度低时介入做语义校正；
    VLM 成功后重置 SUTrack 模板，实现跨帧校准循环。
    """

    def __init__(self, params, dataset_name):
        super().__init__(params)
        self.params = params
        self.dataset_name = dataset_name

        # ---- SUTrack 初始化 ----
        network = build_sutrack(params.cfg)
        network.load_state_dict(
            torch.load(params.checkpoint, map_location='cpu')['net'], strict=True
        )
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()

        self.template_list = []
        self.template_anno_list = []
        self.num_template = self.cfg.TEST.NUM_TEMPLATES

        self.fx_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.ENCODER.STRIDE
        if self.cfg.TEST.WINDOW:
            self.output_window = hann2d(
                torch.tensor([self.fx_sz, self.fx_sz]).long(), centered=True
            ).cuda()
        self.task_index_batch = None

        # SUTrack 在线更新配置
        DNAME = dataset_name.upper()
        self.update_intervals = (
            self.cfg.TEST.UPDATE_INTERVALS[DNAME]
            if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DNAME)
            else self.cfg.TEST.UPDATE_INTERVALS.DEFAULT
        )
        self.update_threshold = (
            self.cfg.TEST.UPDATE_THRESHOLD[DNAME]
            if hasattr(self.cfg.TEST.UPDATE_THRESHOLD, DNAME)
            else self.cfg.TEST.UPDATE_THRESHOLD.DEFAULT
        )

        # ---- VLM 初始化 ----
        self.vlm = VLMEngine(params)
        self.vlm_mode = getattr(params, 'vlm_mode', 'visual')       # 'visual' | 'cognitive'
        self.trigger_mode = getattr(params, 'trigger_mode', 'keyframe')  # 'keyframe' | 'confidence' | 'hybrid'
        self.conf_threshold = getattr(params, 'conf_threshold', 0.3)  # 置信度触发阈值

        # Visual 模式配置
        self.num_frames = getattr(params, 'num_frames', 2)
        self.prompt_name = getattr(params, 'prompt_name',
                                   'two_image' if self.num_frames == 2 else 'three_image')

        # Cognitive 模式配置
        self.track_prompt = getattr(params, 'track_prompt', 'cognitive')
        self.init_prompt = getattr(params, 'init_prompt', 'init_memory')
        self.memory: Dict = {"appearance": "", "motion": "", "context": "", "last_update": 0}

        # ---- 公共跟踪状态 ----
        self.state: Optional[List[float]] = None
        self.init_image: Optional[np.ndarray] = None
        self.init_bbox: Optional[List[float]] = None
        self.prev_image: Optional[np.ndarray] = None
        self.prev_bbox: Optional[List[float]] = None
        self.language_description: str = "the target object"

        # 统计
        self.frame_id = 0
        self.vlm_call_count = 0
        self.seq_name = None

        self.debug = getattr(params, 'debug', 0)

    # ============================================================
    # 初始化
    # ============================================================

    def initialize(self, image, info: dict):
        """初始化 SUTrack 模板 + VLM 状态"""
        self.frame_id = 0
        self.vlm_call_count = 0
        self.seq_name = info.get('seq_name', 'unknown')

        # 公共状态
        self.state = list(info['init_bbox'])
        self.init_image = image.copy()
        self.init_bbox = list(info['init_bbox'])
        self.prev_image = image.copy()
        self.prev_bbox = list(info['init_bbox'])
        lang = info.get('init_nlp', None)
        self.language_description = lang if lang else "the target object"

        # 初始化 SUTrack
        self._sutrack_init(image, info)

        # Cognitive模式：生成初始记忆
        if self.vlm_mode == 'cognitive':
            self._generate_initial_memory(image, self.state)

        if self.debug >= 1:
            print(f"[Hybrid] Init '{self.seq_name}' "
                  f"vlm={self.vlm_mode} trigger={self.trigger_mode}")

    def _sutrack_init(self, image, info):
        """初始化 SUTrack 模板"""
        z_patch, resize_factor = sample_target(
            image, info['init_bbox'],
            self.params.template_factor,
            output_sz=self.params.template_size
        )
        template = self.preprocessor.process(z_patch)
        self.template_list = [template] * self.num_template

        prev_box_crop = transform_image_to_crop(
            torch.tensor(info['init_bbox']),
            torch.tensor(info['init_bbox']),
            resize_factor,
            torch.Tensor([self.params.template_size, self.params.template_size]),
            normalize=True
        )
        self.template_anno_list = [prev_box_crop.to(template.device).unsqueeze(0)] * self.num_template

        # 语言分支（若SUTrack支持）
        self.text_src = None
        if getattr(self.cfg.TEST, 'MULTI_MODAL_LANGUAGE', False):
            import clip
            init_nlp = info.get('init_nlp', None)
            if init_nlp:
                nlp_ids = clip.tokenize(init_nlp).squeeze(0)
            else:
                nlp_ids = torch.zeros(77, dtype=torch.long)
            text_data = nlp_ids.unsqueeze(0).to(template.device)
            with torch.no_grad():
                self.text_src = self.network.forward_textencoder(text_data=text_data)

    def _generate_initial_memory(self, image, bbox):
        """Cognitive模式：生成初始语义记忆，使用parse_memory_state统一解析"""
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
            if self.debug >= 1:
                print(f"[Hybrid] Memory init: {self.memory['appearance'][:60]}...")
        except Exception as e:
            if self.debug >= 1:
                print(f"[Hybrid] Memory init failed ({e}), using description fallback")
            self.memory = {
                "appearance": self.language_description,
                "motion": "unknown", "context": "unknown", "last_update": 0
            }

    # ============================================================
    # 跟踪主循环
    # ============================================================

    def track(self, image, info: dict = None):
        """每帧跟踪：SUTrack 基础 + VLM 按需校正"""
        self.frame_id += 1
        H, W = image.shape[:2]

        # --- Step 1: SUTrack 跟踪 ---
        sutrack_bbox, conf_score = self._sutrack_track(image)

        # --- Step 2: 判断是否触发 VLM ---
        is_keyframe = info.get('is_keyframe', False) if info else False
        trigger_vlm = self._should_trigger_vlm(is_keyframe, conf_score)

        vlm_triggered = False
        if trigger_vlm:
            vlm_result = self._vlm_infer(image, W, H)
            if vlm_result is not None:
                # VLM校正成功 → 采用VLM结果，重置SUTrack
                self.state = vlm_result
                self._sutrack_update_template(image, vlm_result)
                self.prev_image = image.copy()
                self.prev_bbox = vlm_result
                vlm_triggered = True
                self.vlm_call_count += 1
                if self.debug >= 1:
                    print(f"[Hybrid] F{self.frame_id}: VLM校正 conf={conf_score:.3f} "
                          f"bbox={[round(v,1) for v in vlm_result]}")
            else:
                # VLM解析失败 → last-frame fallback（保留 SUTrack 结果）
                self.state = sutrack_bbox
                if self.debug >= 1:
                    print(f"[Hybrid] F{self.frame_id}: VLM parse failed, using SUTrack")
        else:
            # 不触发VLM → 直接用SUTrack结果
            self.state = sutrack_bbox

        # --- Step 3: SUTrack 模板在线更新（仅未触发VLM时） ---
        if not vlm_triggered and self.num_template > 1:
            if (self.frame_id % self.update_intervals == 0) and (conf_score > self.update_threshold):
                self._sutrack_update_template(image, self.state)

        return {
            "target_bbox": self.state,
            "best_score": conf_score,
        }

    def _should_trigger_vlm(self, is_keyframe: bool, conf_score: float) -> bool:
        """判断是否满足VLM触发条件"""
        kf_trigger = is_keyframe
        conf_trigger = conf_score < self.conf_threshold
        if self.trigger_mode == 'keyframe':
            return kf_trigger
        elif self.trigger_mode == 'confidence':
            return conf_trigger
        else:  # 'hybrid'
            return kf_trigger or conf_trigger

    # ============================================================
    # SUTrack 跟踪相关
    # ============================================================

    def _sutrack_track(self, image):
        """运行 SUTrack 推理，返回 (bbox_xywh, conf_score)"""
        x_patch, resize_factor = sample_target(
            image, self.state,
            self.params.search_factor,
            output_sz=self.params.search_size
        )
        search = self.preprocessor.process(x_patch)

        with torch.no_grad():
            enc_opt = self.network.forward_encoder(
                self.template_list, [search],
                self.template_anno_list, self.text_src, self.task_index_batch
            )
            out_dict = self.network.forward_decoder(feature=enc_opt)

        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map if self.cfg.TEST.WINDOW else pred_score_map

        H_img, W_img = image.shape[:2]
        if 'size_map' in out_dict:
            pred_boxes, conf_score = self.network.decoder.cal_bbox(
                response, out_dict['size_map'], out_dict['offset_map'], return_score=True
            )
        else:
            pred_boxes, conf_score = self.network.decoder.cal_bbox(
                response, out_dict['offset_map'], return_score=True
            )

        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
        bbox = clip_box(self._map_box_back(pred_box, resize_factor), H_img, W_img, margin=10)
        return bbox, conf_score

    def _map_box_back(self, pred_box, resize_factor):
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def _sutrack_update_template(self, image, bbox):
        """用新的 bbox 更新 SUTrack 模板"""
        z_patch, resize_factor = sample_target(
            image, bbox,
            self.params.template_factor,
            output_sz=self.params.template_size
        )
        template = self.preprocessor.process(z_patch)
        self.template_list.append(template)
        if len(self.template_list) > self.num_template:
            self.template_list.pop(1)

        prev_box_crop = transform_image_to_crop(
            torch.tensor(bbox), torch.tensor(bbox),
            resize_factor,
            torch.Tensor([self.params.template_size, self.params.template_size]),
            normalize=True
        )
        self.template_anno_list.append(prev_box_crop.to(template.device).unsqueeze(0))
        if len(self.template_anno_list) > self.num_template:
            self.template_anno_list.pop(1)

    # ============================================================
    # VLM 推理相关
    # ============================================================

    def _vlm_infer(self, image, W, H) -> Optional[List[float]]:
        """
        构建VLM输入并推理，返回 bbox_xywh 或 None。
        根据 vlm_mode 选择 visual 或 cognitive 范式。
        """
        try:
            if self.vlm_mode == 'cognitive':
                return self._vlm_infer_cognitive(image, W, H)
            else:
                return self._vlm_infer_visual(image, W, H)
        except Exception as e:
            if self.debug >= 1:
                print(f"[Hybrid] VLM error: {e}")
            return None

    def _vlm_infer_visual(self, image, W, H) -> Optional[List[float]]:
        """Visual VLM推理（纯图像输入）"""
        init_with_box = draw_bbox(self.init_image, self.init_bbox, color=(0, 255, 0))

        if self.num_frames == 3:
            prev_with_box = draw_bbox(self.prev_image, self.prev_bbox, color=(0, 0, 255))
            images = [init_with_box, prev_with_box, image]
        else:
            images = [init_with_box, image]

        prompt = get_prompt(self.prompt_name, target_description=self.language_description)
        output = self.vlm.infer(images, prompt)
        bbox_xyxy = parse_bbox_from_text(output, W, H)
        return xyxy_to_xywh(bbox_xyxy) if bbox_xyxy is not None else None

    def _vlm_infer_cognitive(self, image, W, H) -> Optional[List[float]]:
        """Cognitive VLM推理（语义记忆 + 图像输入）"""
        prev_with_box = draw_bbox(self.prev_image, self.prev_bbox, color=(0, 0, 255))
        prompt = get_prompt(
            self.track_prompt,
            memory_appearance=self.memory['appearance'],
            memory_motion=self.memory['motion'],
            memory_context=self.memory['context']
        )
        output = self.vlm.infer([prev_with_box, image], prompt)

        bbox_xyxy = parse_bbox_from_text(output, W, H)
        if bbox_xyxy is None:
            return None

        # 同步更新记忆
        new_memory = parse_memory_state(output)
        if new_memory:
            new_memory['last_update'] = self.frame_id
            self.memory = new_memory

        return xyxy_to_xywh(bbox_xyxy)

    # ============================================================
    # 兼容性工具
    # ============================================================

    def extract_token_from_nlp_clip(self, nlp):
        import clip
        if nlp is None:
            return torch.zeros(77, dtype=torch.long), torch.zeros(77, dtype=torch.long)
        nlp_ids = clip.tokenize(nlp).squeeze(0)
        nlp_masks = (nlp_ids == 0).long()
        return nlp_ids, nlp_masks


def get_tracker_class():
    return VLMHybrid

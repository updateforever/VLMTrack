"""
Qwen3VL Three-Image Tracker
改进方案: 使用初始帧作为固定锚点,避免逐帧漂移

跟踪范式:
- Image 1: 初始帧 + 初始框 (绿色) - 固定参考
- Image 2: 上一帧 + 预测框 (蓝色) - 短期运动
- Image 3: 当前帧 - 待预测

优势:
- 稳定锚点: 初始帧始终提供正确目标信息
- 运动连续性: 上一帧提供短期运动线索
- 双重约束: 长期+短期信息结合,减少漂移
"""
from lib.test.tracker.basetracker import BaseTracker
from lib.test.evaluation.environment import env_settings
import torch
import cv2
import numpy as np
import re
import json
import os
import time
import base64
from pathlib import Path
from PIL import Image
from typing import List, Optional, Tuple, Set

# 复用原始tracker的工具函数
from lib.test.tracker.qwen3vl import (
    extract_bbox_from_model_output,
    xyxy_to_xywh,
    xywh_to_xyxy,
    numpy_to_base64,
    qwen3vl_api_chat,
    read_keyframe_indices
)


class QWEN3VL_ThreeImage(BaseTracker):
    """
    三图跟踪模式: 初始帧 + 上一帧 + 当前帧
    
    改进点:
    - 固定初始帧作为目标锚点
    - 上一帧提供运动连续性
    - 减少逐帧漂移问题
    """
    
    def __init__(self, params, dataset_name):
        super(QWEN3VL_ThreeImage, self).__init__(params)
        self.params = params
        self.dataset_name = dataset_name
        
        # 推理模式
        self.mode = getattr(params, 'mode', 'local')
        
        # 加载模型/配置API
        if self.mode == 'local':
            self._load_local_model()
        else:
            self._setup_api()
        
        # State
        self.state = None
        
        # 三图跟踪状态
        self.init_image = None       # 固定: 初始帧
        self.init_bbox = None        # 固定: 初始框
        self.prev_image = None       # 动态: 上一帧
        self.prev_bbox = None        # 动态: 上一帧框
        
        self.language_description = None
        self.frame_id = 0
        self.seq_name = None
        
        # Keyframe Tracking
        self.keyframe_indices = None
        self.keyframe_results = {}
        self.use_keyframe = getattr(params, 'use_keyframe', False)
        self.keyframe_root = getattr(params, 'keyframe_root', None)
        
        # Debug
        self.debug = getattr(params, 'debug', 0)
        self.vis_dir = None
    
    def _load_local_model(self):
        """加载本地模型"""
        from transformers import AutoProcessor
        
        model_path = getattr(self.params, 'model_path', None)
        model_name = getattr(self.params, 'model_name', 'Qwen/Qwen3-VL-4B-Instruct')
        
        actual_path = model_path or model_name
        print(f"[ThreeImage] Loading local model: {actual_path}")
        
        if 'qwen3' in actual_path.lower():
            from transformers import Qwen3VLForConditionalGeneration
            model_class = Qwen3VLForConditionalGeneration
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model_class = Qwen2_5_VLForConditionalGeneration
        
        try:
            self.model = model_class.from_pretrained(
                actual_path, torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2", device_map="auto"
            )
        except Exception:
            self.model = model_class.from_pretrained(
                actual_path, torch_dtype=torch.bfloat16, device_map="auto"
            )
        
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(actual_path)
        print(f"[ThreeImage] Model loaded")
    
    def _setup_api(self):
        """配置API"""
        self.api_model = getattr(self.params, 'api_model', 'qwen3-vl-235b-a22b-instruct')
        self.api_base_url = getattr(self.params, 'api_base_url', 
                                     'https://dashscope.aliyuncs.com/compatible-mode/v1')
        self.api_key = getattr(self.params, 'api_key', os.environ.get('DASHSCOPE_API_KEY', ''))
        
        print(f"[ThreeImage] API mode: {self.api_model}")
    
    def _draw_bbox_on_image(self, image: np.ndarray, bbox_xywh: List[float], 
                            color=(0, 255, 0), thickness=3) -> np.ndarray:
        """在图像上绘制bbox"""
        img = image.copy()
        x, y, w, h = [int(v) for v in bbox_xywh]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        return img
    
    def _build_tracking_prompt(self, description: str) -> str:
        """
        构建三图跟踪prompt
        输入: 3张图片
          - 图1: 初始帧 + 绿框 (ground truth, 固定锐点)
          - 图2: 上一帧 + 蓝框 (历史预测结果, 可能不准)
          - 图3: 当前帧 (待预测)
        """
        return (
            # 第一张图: 初始帧作为固定参考 (ground truth)
            f"The first image shows the initial frame with the target object marked by a green bounding box (ground truth). "
            f"The target is: {description}. "
            # 第二张图: 上一帧的预测框,可能不准确
            f"The second image shows the previous frame with the predicted target location marked by a blue bounding box (may not be accurate). "
            # 第三张图: 当前帧需要定位
            f"The third image is the current frame. "
            # 任务: 以初始帧为准,参考上一帧运动
            f"Locate the same target in the third image based on the ground truth (image 1), "
            f"using the previous frame (image 2) only as motion reference. "
            f"Output its bbox coordinates using JSON format."
        )
    
    def _run_inference(self, init_img: np.ndarray, prev_img: np.ndarray, 
                      current_img: np.ndarray, prompt: str) -> str:
        """运行三图推理"""
        if self.mode == 'api':
            # API模式
            init_b64 = numpy_to_base64(init_img)
            prev_b64 = numpy_to_base64(prev_img)
            curr_b64 = numpy_to_base64(current_img)
            return qwen3vl_api_chat(
                images_b64=[init_b64, prev_b64, curr_b64],
                prompt=prompt,
                model_name=self.api_model,
                base_url=self.api_base_url,
                api_key=self.api_key,
            )
        else:
            return self._run_local_inference(init_img, prev_img, current_img, prompt)
    
    def _run_local_inference(self, init_img: np.ndarray, prev_img: np.ndarray,
                            current_img: np.ndarray, prompt: str) -> str:
        """本地三图推理"""
        init_pil = Image.fromarray(init_img)
        prev_pil = Image.fromarray(prev_img)
        curr_pil = Image.fromarray(current_img)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": init_pil},
                    {"type": "image", "image": prev_pil},
                    {"type": "image", "image": curr_pil},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
    
    def _save_visualization(self, init_with_box: np.ndarray, prev_with_box: np.ndarray,
                           search_img: np.ndarray, pred_bbox_xywh: List[float], frame_id: int):
        """保存三图可视化"""
        if self.debug < 2 or self.vis_dir is None:
            return
        
        # 在搜索图上画预测框
        result_img = self._draw_bbox_on_image(search_img, pred_bbox_xywh, color=(255, 0, 0), thickness=3)
        
        # 调整高度一致
        h1, w1 = init_with_box.shape[:2]
        h2, w2 = prev_with_box.shape[:2]
        h3, w3 = result_img.shape[:2]
        
        target_h = max(h1, h2, h3)
        
        def resize_to_height(img, target_h):
            h, w = img.shape[:2]
            if h != target_h:
                scale = target_h / h
                new_w = int(w * scale)
                return cv2.resize(img, (new_w, target_h))
            return img
        
        init_resized = resize_to_height(init_with_box, target_h)
        prev_resized = resize_to_height(prev_with_box, target_h)
        result_resized = resize_to_height(result_img, target_h)
        
        # 水平拼接三图
        combined = np.hstack([init_resized, prev_resized, result_resized])
        
        # 添加标注
        cv2.putText(combined, f"Frame {frame_id}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Init (Green)", (10, target_h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(combined, "Prev (Blue)", (init_resized.shape[1] + 10, target_h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(combined, "Current (Red)", (init_resized.shape[1] + prev_resized.shape[1] + 10, target_h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 保存
        vis_path = os.path.join(self.vis_dir, f"{self.seq_name}_{frame_id:04d}.jpg")
        cv2.imwrite(vis_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        
        # 实时显示
        if self.debug >= 3:
            cv2.imshow('Three-Image Tracking', cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
    
    def initialize(self, image, info: dict):
        """初始化跟踪器"""
        self.frame_id = 0
        H, W = image.shape[:2]
        
        self.state = list(info['init_bbox'])
        
        # 三图跟踪: 保存初始帧和框(固定)
        self.init_image = image.copy()
        self.init_bbox = list(info['init_bbox'])
        
        # 初始化上一帧(动态)
        self.prev_image = image.copy()
        self.prev_bbox = list(info['init_bbox'])
        
        self.seq_name = info.get('seq_name', None)
        
        self.language_description = info.get('init_nlp', None)
        if not self.language_description:
            self.language_description = "the target object marked in green box"
        
        # 关键帧跟踪
        if self.use_keyframe and self.keyframe_root:
            self.keyframe_indices = read_keyframe_indices(self.keyframe_root, self.seq_name)
            if self.keyframe_indices:
                print(f"[ThreeImage] Keyframe mode: {len(self.keyframe_indices)} keyframes")
        
        # 可视化
        if self.debug >= 2:
            env = env_settings()
            self.vis_dir = os.path.join(env.results_path, 'qwen3vl_three_image', 'vis', 
                                       self.seq_name or 'unknown')
            os.makedirs(self.vis_dir, exist_ok=True)
        
        if self.debug >= 1:
            mode_str = f"{self.mode} + three-image"
            if self.use_keyframe:
                mode_str += " + keyframe"
            print(f"[ThreeImage] Initialize: bbox={self.state}, mode={mode_str}")
    
    def track(self, image, info: dict = None):
        """三图跟踪"""
        self.frame_id += 1
        H, W = image.shape[:2]
        
        # 稀疏跟踪: 非关键帧跳过
        if self.use_keyframe and self.keyframe_indices:
            if self.frame_id not in self.keyframe_indices:
                if self.debug >= 1:
                    print(f"[ThreeImage] Frame {self.frame_id}: Skipped (non-keyframe)")
                return None
        
        try:
            # 准备三张图
            init_with_box = self._draw_bbox_on_image(
                self.init_image, self.init_bbox, color=(0, 255, 0), thickness=3
            )
            prev_with_box = self._draw_bbox_on_image(
                self.prev_image, self.prev_bbox, color=(0, 0, 255), thickness=3
            )
            current_img = image
            
            # 构建prompt
            prompt = self._build_tracking_prompt(self.language_description)
            
            # 三图推理
            output_text = self._run_inference(init_with_box, prev_with_box, current_img, prompt)
            
            if self.debug >= 1:
                kf_tag = "[KF]" if self.use_keyframe else ""
                print(f"[ThreeImage] Frame {self.frame_id}{kf_tag}: {output_text[:100]}...")
            
            # 解析bbox
            bbox_xyxy = extract_bbox_from_model_output(output_text, W, H)
            
            if bbox_xyxy is not None:
                pred_bbox = xyxy_to_xywh(bbox_xyxy)
                self.state = pred_bbox
                
                # 可视化
                self._save_visualization(init_with_box, prev_with_box, image, pred_bbox, self.frame_id)
                
                # 更新上一帧(短期记忆)
                self.prev_image = image.copy()
                self.prev_bbox = pred_bbox
                
                # 缓存关键帧
                if self.use_keyframe:
                    self.keyframe_results[self.frame_id] = pred_bbox
            else:
                if self.debug >= 1:
                    print(f"[ThreeImage] Frame {self.frame_id}: Failed to parse bbox")
                self.state = [0, 0, 0, 0]
                
        except Exception as e:
            print(f"[ThreeImage] Error frame {self.frame_id}: {e}")
            self.state = [0, 0, 0, 0]
        
        return {"target_bbox": self.state}


def get_tracker_class():
    return QWEN3VL_ThreeImage

"""
Qwen3VL Hybrid Tracker: Three-Image + Memory Bank
终极方案: 结合三图跟踪和记忆库的优势

跟踪范式:
- Image 1: 初始帧 + 初始框 (绿色) - 固定视觉锚点
- Image 2: 上一帧 + 预测框 (蓝色) - 短期运动
- Image 3: 当前帧 - 待预测
- Memory: 语义描述 - 文本锚点

优势:
- 双重锚定: 视觉(初始帧) + 语义(记忆库)
- 运动连续: 上一帧提供短期线索
- 自适应: 记忆库随目标变化更新
- 最强鲁棒性: 多重约束减少漂移
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
from pathlib import Path
from PIL import Image
from typing import List, Optional, Dict

# 复用工具函数
from lib.test.tracker.qwen3vl import (
    extract_bbox_from_model_output,
    xyxy_to_xywh,
    numpy_to_base64,
    qwen3vl_api_chat,
    read_keyframe_indices
)


class QWEN3VL_Hybrid(BaseTracker):
    """
    混合跟踪模式: 三图 + 记忆库
    
    最强配置:
    - 视觉锚点: 初始帧
    - 运动线索: 上一帧
    - 语义引导: 记忆库
    """
    
    def __init__(self, params, dataset_name):
        super(QWEN3VL_Hybrid, self).__init__(params)
        self.params = params
        self.dataset_name = dataset_name
        
        # 推理模式
        self.mode = getattr(params, 'mode', 'api')
        
        if self.mode == 'local':
            self._load_local_model()
        else:
            self._setup_api()
        
        # State
        self.state = None
        
        # 三图状态
        self.init_image = None
        self.init_bbox = None
        self.prev_image = None
        self.prev_bbox = None
        
        # 记忆库
        self.memory = {
            "appearance": "",
            "motion": "",
            "context": "",
            "last_update": 0
        }
        self.memory_update_interval = getattr(params, 'memory_update_interval', 10)
        
        self.language_description = None
        self.frame_id = 0
        self.seq_name = None
        
        # Keyframe
        self.keyframe_indices = None
        self.keyframe_results = {}
        self.use_keyframe = getattr(params, 'use_keyframe', False)
        self.keyframe_root = getattr(params, 'keyframe_root', None)
        
        # Debug
        self.debug = getattr(params, 'debug', 0)
        self.vis_dir = None
    
    def _load_local_model(self):
        """加载本地模型"""
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        
        model_name = getattr(self.params, 'model_name', 'Qwen/Qwen3-VL-4B-Instruct')
        print(f"[Hybrid] Loading: {model_name}")
        
        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2", device_map="auto"
            )
        except:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto"
            )
        
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
    
    def _setup_api(self):
        """配置API"""
        self.api_model = getattr(self.params, 'api_model', 'qwen3-vl-plus-2025-09-23')
        self.api_base_url = getattr(self.params, 'api_base_url', 'http://10.128.202.100:3010/v1')
        self.api_key = getattr(self.params, 'api_key', os.environ.get('DASHSCOPE_API_KEY', ''))
        print(f"[Hybrid] API: {self.api_model}")
    
    def _draw_bbox(self, image: np.ndarray, bbox: List[float], color=(0,255,0), thickness=3) -> np.ndarray:
        """绘制bbox"""
        img = image.copy()
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)
        return img
    
    def _generate_memory_prompt(self) -> str:
        """生成记忆prompt (与 memory tracker相同)"""
        return (
            "Analyze the target object marked by the green bounding box. "
            "Provide a detailed description in JSON format: "
            '{"appearance": "color, shape, texture, features", '
            '"motion": "current motion state", '
            '"context": "surrounding objects and position"}. '
            "Be specific and focus on distinctive features. Output ONLY the JSON object."
        )
    
    def _tracking_prompt(self) -> str:
        """
        三图+记忆库跟踪prompt
        输入: 3张图片 + 记忆库
          - 图1: 初始帧 + 绿框 (ground truth, 视觉锐点)
          - 图2: 上一帧 + 蓝框 (历史预测, 可能不准)
          - 图3: 当前帧 (待预测)
          - 记忆: 目标语义描述 (语义锐点)
        """
        return (
            # 提供记忆库作为语义锐点
            f"Target memory: appearance is {self.memory['appearance']}, "
            f"motion is {self.memory['motion']}, "
            f"context is {self.memory['context']}. "
            # 第一张图: 初始帧作为视觉锐点 (ground truth)
            f"The first image shows the initial frame with the target marked by a green bounding box (ground truth). "
            # 第二张图: 上一帧的预测框,仅供运动参考
            f"The second image shows the previous frame with the predicted target location marked by a blue bounding box (may not be accurate, use only for motion reference). "
            # 第三张图: 当前帧需要定位
            f"The third image is the current frame. "
            # 任务: 结合记忆(语义)、初始帧(视觉)、上一帧(运动)
            f"Locate the target that matches both the memory description and the ground truth appearance (image 1), "
            f"using the previous frame (image 2) only for motion reference. "
            f"Output its bbox coordinates in the third image using JSON format."
        )
    
    def _run_inference(self, images: List[np.ndarray], prompt: str) -> str:
        """推理"""
        if self.mode == 'api':
            images_b64 = [numpy_to_base64(img) for img in images]
            return qwen3vl_api_chat(
                images_b64=images_b64, prompt=prompt,
                model_name=self.api_model,
                base_url=self.api_base_url,
                api_key=self.api_key
            )
        else:
            return self._run_local_inference(images, prompt)
    
    def _run_local_inference(self, images: List[np.ndarray], prompt: str) -> str:
        """本地推理"""
        images_pil = [Image.fromarray(img) for img in images]
        content = [{"type": "image", "image": img} for img in images_pil]
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )[0]
    
    def _parse_memory(self, text: str) -> Dict:
        """解析记忆JSON"""
        try:
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            data = json.loads(text.strip())
            
            # 将字典值转为字符串
            def dict_to_str(val):
                if isinstance(val, dict):
                    return ', '.join(f"{k}: {v}" for k, v in val.items())
                return str(val)
            
            return {
                "appearance": dict_to_str(data.get("appearance", "")),
                "motion": dict_to_str(data.get("motion", "")),
                "context": dict_to_str(data.get("context", "")),
                "last_update": self.frame_id
            }
        except:
            return self.memory
    
    def _generate_memory(self, image: np.ndarray, bbox: List[float]) -> Dict:
        """生成记忆"""
        img_with_box = self._draw_bbox(image, bbox, (0,255,0))
        prompt = self._generate_memory_prompt()
        output = self._run_inference([img_with_box], prompt)
        
        if self.debug >= 1:
            print(f"[Hybrid] Memory: {output[:150]}...")
        
        return self._parse_memory(output)
    
    def _save_visualization(self, prev_with_box: np.ndarray, search_img: np.ndarray, 
                           pred_bbox_xywh: List[float], frame_id: int):
        """保存混合模式可视化 (记忆+三图)"""
        if self.debug < 2 or self.vis_dir is None:
            return
        
        result_img = self._draw_bbox(search_img, pred_bbox_xywh, (255,0,0), 3)
        h1, w1 = prev_with_box.shape[:2]
        h2, w2 = result_img.shape[:2]
        target_h = max(h1, h2)
        
        def resize(img, h):
            return cv2.resize(img, (int(img.shape[1]*h/img.shape[0]), h)) if img.shape[0]!=h else img
        
        prev_resized = resize(prev_with_box, target_h)
        result_resized = resize(result_img, target_h)
        
        # 记忆显示区域
        memory_img = np.ones((target_h, 400, 3), dtype=np.uint8) * 255
        y, font = 30, cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(memory_img, "Memory Bank:", (10,y), font, 0.7, (0,0,255), 2)
        y += 35
        
        app = self.memory.get('appearance','')[:80]
        cv2.putText(memory_img, "Appearance:", (10,y), font, 0.5, (0,0,0), 1)
        y += 25
        for line in [app[i:i+40] for i in range(0,len(app),40)][:2]:
            cv2.putText(memory_img, line, (15,y), font, 0.4, (0,0,0), 1)
            y += 20
        
        cv2.putText(memory_img, f"Motion: {self.memory.get('motion','')[:60]}", (10,y), font, 0.4, (0,0,0), 1)
        y += 25
        cv2.putText(memory_img, f"Context: {self.memory.get('context','')[:60]}", (10,y), font, 0.4, (0,0,0), 1)
        y += 35
        cv2.putText(memory_img, f"Update: Frame {self.memory.get('last_update',0)}", (10,y), font, 0.4, (0,100,0), 1)
        
        combined = np.hstack([memory_img, prev_resized, result_resized])
        cv2.putText(combined, f"Frame {frame_id} [HYBRID]", (410,30), font, 1, (0,255,0), 2)
        cv2.putText(combined, "Prev (Blue)", (410, target_h-10), font, 0.6, (255,0,0), 2)
        cv2.putText(combined, "Current (Red)", (410+prev_resized.shape[1], target_h-10), font, 0.6, (0,0,255), 2)
        
        vis_path = os.path.join(self.vis_dir, f"{self.seq_name}_{frame_id:04d}.jpg")
        cv2.imwrite(vis_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        
        if self.debug >= 3:
            cv2.imshow('Hybrid Tracking', cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
    
    def initialize(self, image, info: dict):
        """初始化"""
        self.frame_id = 0
        self.state = list(info['init_bbox'])
        
        # 三图状态
        self.init_image = image.copy()
        self.init_bbox = list(info['init_bbox'])
        self.prev_image = image.copy()
        self.prev_bbox = list(info['init_bbox'])
        
        self.seq_name = info.get('seq_name', None)
        self.language_description = info.get('init_nlp', None) or "the target object"
        
        # 生成初始记忆
        print(f"[Hybrid] Generating memory...")
        self.memory = self._generate_memory(image, self.state)
        
        if self.debug >= 1:
            print(f"[Hybrid] Appearance: {self.memory['appearance'][:80]}...")
        
        # 关键帧
        if self.use_keyframe and self.keyframe_root:
            self.keyframe_indices = read_keyframe_indices(self.keyframe_root, self.seq_name)
            if self.keyframe_indices:
                print(f"[Hybrid] Keyframes: {len(self.keyframe_indices)}")
        
        # 可视化
        if self.debug >= 2:
            env = env_settings()
            self.vis_dir = os.path.join(env.results_path, 'qwen3vl_hybrid', 'vis', 
                                       self.seq_name or 'unknown')
            os.makedirs(self.vis_dir, exist_ok=True)
        
        mode_str = f"{self.mode} + three-image + memory"
        if self.use_keyframe:
            mode_str += " + keyframe"
        print(f"[Hybrid] Init: {mode_str}")
    
    def track(self, image, info: dict = None):
        """混合跟踪"""
        self.frame_id += 1
        H, W = image.shape[:2]
        
        # 稀疏跟踪
        if self.use_keyframe and self.keyframe_indices:
            if self.frame_id not in self.keyframe_indices:
                if self.debug >= 1:
                    print(f"[Hybrid] Frame {self.frame_id}: Skipped")
                return None
        
        try:
            # 准备三图
            init_with_box = self._draw_bbox(self.init_image, self.init_bbox, (0,255,0))
            prev_with_box = self._draw_bbox(self.prev_image, self.prev_bbox, (0,0,255))
            
            # 三图+记忆库推理
            prompt = self._tracking_prompt()
            output = self._run_inference([init_with_box, prev_with_box, image], prompt)
            
            if self.debug >= 1:
                kf = "[KF]" if self.use_keyframe else ""
                print(f"[Hybrid] Frame {self.frame_id}{kf}: {output[:80]}...")
            
            # 解析
            bbox_xyxy = extract_bbox_from_model_output(output, W, H)
            
            if bbox_xyxy:
                pred_bbox = xyxy_to_xywh(bbox_xyxy)
                self.state = pred_bbox
                
                # 可视化
                self._save_visualization(prev_with_box, image, pred_bbox, self.frame_id)
                
                # 更新上一帧
                self.prev_image = image.copy()
                self.prev_bbox = pred_bbox
                
                # 更新记忆
                if (self.frame_id - self.memory['last_update']) >= self.memory_update_interval:
                    if self.debug >= 1:
                        print(f"[Hybrid] Updating memory...")
                    self.memory = self._generate_memory(image, pred_bbox)
                
                # 缓存
                if self.use_keyframe:
                    self.keyframe_results[self.frame_id] = pred_bbox
            else:
                if self.debug >= 1:
                    print(f"[Hybrid] Frame {self.frame_id}: Parse failed")
                self.state = [0,0,0,0]
        
        except Exception as e:
            print(f"[Hybrid] Error {self.frame_id}: {e}")
            self.state = [0,0,0,0]
        
        return {"target_bbox": self.state}


def get_tracker_class():
    return QWEN3VL_Hybrid

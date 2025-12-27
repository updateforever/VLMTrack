"""
Qwen3VL Memory Bank Tracker
改进方案: VLM维护目标的语义记忆库,提供持续的目标描述引导

跟踪范式:
- 初始化: 生成目标的详细语义描述(记忆库)
- 跟踪: 使用记忆库 + 上一帧 → 预测当前帧
- 更新: 定期更新记忆库(关键帧)

优势:
- 语义锚定: 文本描述提供稳定的目标特征
- 自适应: 记忆库随目标外观变化更新
- 鲁棒性: 即使短期跟丢,也能通过语义重新定位
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
from typing import List, Optional, Tuple, Set, Dict

# 复用工具函数
from lib.test.tracker.qwen3vl import (
    extract_bbox_from_model_output,
    xyxy_to_xywh,
    xywh_to_xyxy,
    numpy_to_base64,
    qwen3vl_api_chat,
    read_keyframe_indices
)


class QWEN3VL_Memory(BaseTracker):
    """
    记忆库跟踪模式: VLM维护目标的语义记忆
    
    改进点:
    - 语义记忆: 文本描述目标特征
    - 定期更新: 关键帧更新记忆
    - 长期稳定: 减少短期噪声影响
    """
    
    def __init__(self, params, dataset_name):
        super(QWEN3VL_Memory, self).__init__(params)
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
        self.prev_image = None
        self.prev_bbox = None
        
        # 记忆库
        self.memory = {
            "appearance": "",      # 外观描述
            "motion": "",          # 运动状态
            "context": "",         # 场景上下文
            "last_update": 0,      # 上次更新帧
            "confidence": 1.0      # 记忆置信度
        }
        
        self.memory_update_interval = getattr(params, 'memory_update_interval', 10)
        self.memory_confidence_threshold = getattr(params, 'memory_confidence_threshold', 0.8)
        
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
        print(f"[Memory] Loading local model: {actual_path}")
        
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
        print(f"[Memory] Model loaded")
    
    def _setup_api(self):
        """配置API"""
        self.api_model = getattr(self.params, 'api_model', 'qwen3-vl-235b-a22b-instruct')
        self.api_base_url = getattr(self.params, 'api_base_url', 
                                     'https://dashscope.aliyuncs.com/compatible-mode/v1')
        self.api_key = getattr(self.params, 'api_key', os.environ.get('DASHSCOPE_API_KEY', ''))
        
        print(f"[Memory] API mode: {self.api_model}")
    
    def _draw_bbox_on_image(self, image: np.ndarray, bbox_xywh: List[float], 
                            color=(0, 255, 0), thickness=3) -> np.ndarray:
        """在图像上绘制bbox"""
        img = image.copy()
        x, y, w, h = [int(v) for v in bbox_xywh]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        return img
    
    def _generate_memory_prompt(self) -> str:
        """生成记忆库的prompt"""
        return (
            "# --- TASK ---\n"
            "Analyze the target object marked by the GREEN box.\n\n"
            
            "# --- OUTPUT ---\n"
            "Provide a detailed description in JSON:\n"
            "{\n"
            '  "appearance": "color, shape, texture, distinctive features",\n'
            '  "motion": "current motion state",\n'
            '  "context": "surrounding objects and position"\n'
            "}\n"
            "Be specific. Output ONLY the JSON object.\n"
        )
    
    def _tracking_with_memory_prompt(self) -> str:
        """使用记忆库跟踪的prompt"""
        return (
            "# --- CORE TASK ---\n"
            "Track the target using semantic memory and motion cues. Determine if target is visible and locate it.\n\n"
            
            "# --- SEMANTIC MEMORY ---\n"
            f"Appearance: {self.memory['appearance']}\n"
            f"Motion: {self.memory['motion']}\n"
            f"Context: {self.memory['context']}\n\n"
            
            "# --- VISUAL REFERENCE ---\n"
            "Image 1 (Previous - BLUE box): Last prediction (may be inaccurate, use only for motion reference).\n"
            "Image 2 (Current): Find the target here.\n\n"
            
            "# --- OUTPUT REQUIREMENT ---\n"
            "Match the target based on: (1) Semantic memory, (2) Motion from Image 1.\n"
            "Output JSON format:\n"
            "{\n"
            '  "bbox": [x1, y1, x2, y2],      // 0-1000 scale. Output [0,0,0,0] if target is invisible/occluded.\n'
            '  "evidence": "Describe matched features from memory and observed motion.",\n'
            '  "confidence": 0.95             // Float between 0.0 (Lost) and 1.0 (Certain).\n'
            "}\n"
        )
    
    def _run_inference(self, images: List[np.ndarray], prompt: str) -> str:
        """运行推理"""
        if self.mode == 'api':
            images_b64 = [numpy_to_base64(img) for img in images]
            return qwen3vl_api_chat(
                images_b64=images_b64,
                prompt=prompt,
                model_name=self.api_model,
                base_url=self.api_base_url,
                api_key=self.api_key,
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
        )
        inputs = inputs.to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
    
    def _parse_memory(self, text: str) -> Dict:
        """解析记忆库JSON"""
        try:
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            
            data = json.loads(text.strip())
            
            # 将字典值转为字符串(如果VLM返回的是字典)
            def dict_to_str(val):
                if isinstance(val, dict):
                    return ', '.join(f"{k}: {v}" for k, v in val.items())
                return str(val)
            
            return {
                "appearance": dict_to_str(data.get("appearance", "")),
                "motion": dict_to_str(data.get("motion", "")),
                "context": dict_to_str(data.get("context", "")),
                "last_update": self.frame_id,
                "confidence": 1.0
            }
        except Exception as e:
            if self.debug >= 1:
                print(f"[Memory] Failed to parse memory: {e}")
            return self.memory
    
    def _generate_memory(self, image: np.ndarray, bbox: List[float]) -> Dict:
        """生成初始记忆库"""
        img_with_box = self._draw_bbox_on_image(image, bbox, color=(0, 255, 0))
        prompt = self._generate_memory_prompt()
        
        output = self._run_inference([img_with_box], prompt)
        
        if self.debug >= 1:
            print(f"[Memory] Generated memory: {output[:200]}...")
        
        return self._parse_memory(output)
    
    def _save_visualization(self, prev_with_box: np.ndarray, search_img: np.ndarray, 
                           pred_bbox_xywh: List[float], frame_id: int):
        """保存记忆库可视化 (包含记忆内容)"""
        if self.debug < 2 or self.vis_dir is None:
            return
        
        result_img = self._draw_bbox_on_image(search_img, pred_bbox_xywh, color=(255, 0, 0), thickness=3)
        
        h1, w1 = prev_with_box.shape[:2]
        h2, w2 = result_img.shape[:2]
        target_h = max(h1, h2)
        
        def resize_to_height(img, target_h):
            h, w = img.shape[:2]
            if h != target_h:
                return cv2.resize(img, (int(w * target_h / h), target_h))
            return img
        
        prev_resized = resize_to_height(prev_with_box, target_h)
        result_resized = resize_to_height(result_img, target_h)
        
        # 创建记忆显示区域
        memory_width = 400
        memory_img = np.ones((target_h, memory_width, 3), dtype=np.uint8) * 255
        
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 标题
        cv2.putText(memory_img, "Memory Bank:", (10, y_offset), font, 0.7, (0, 0, 255), 2)
        y_offset += 35
        
        # 外观
        appearance = self.memory.get('appearance', '')[:80]
        lines = [appearance[i:i+40] for i in range(0, len(appearance), 40)]
        cv2.putText(memory_img, "Appearance:", (10, y_offset), font, 0.5, (0,0,0), 1)
        y_offset += 25
        for line in lines[:2]:
            cv2.putText(memory_img, line, (15, y_offset), font, 0.4, (0,0,0), 1)
            y_offset += 20
        
        # 运动
        motion = self.memory.get('motion', '')[:60]
        cv2.putText(memory_img, f"Motion: {motion}", (10, y_offset), font, 0.4, (0,0,0), 1)
        y_offset += 25
        
        # 上下文
        context = self.memory.get('context', '')[:60]
        cv2.putText(memory_img, f"Context: {context}", (10, y_offset), font, 0.4, (0,0,0), 1)
        y_offset += 35
        
        # 更新信息
        last_update = self.memory.get('last_update', 0)
        cv2.putText(memory_img, f"Last Update: Frame {last_update}", (10, y_offset), font, 0.4, (0,100,0), 1)
        
        # 拼接
        combined = np.hstack([memory_img, prev_resized, result_resized])
        
        # 标注
        cv2.putText(combined, f"Frame {frame_id}", (memory_width + 10, 30), font, 1, (0,255,0), 2)
        cv2.putText(combined, "Prev (Blue)", (memory_width + 10, target_h - 10), font, 0.6, (255,0,0), 2)
        cv2.putText(combined, "Current (Red)", (memory_width + prev_resized.shape[1] + 10, target_h - 10), font, 0.6, (0,0,255), 2)
        
        # 保存
        vis_path = os.path.join(self.vis_dir, f"{self.seq_name}_{frame_id:04d}.jpg")
        cv2.imwrite(vis_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        
        if self.debug >= 3:
            cv2.imshow('Memory-Bank Tracking', cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
    
    def _should_update_memory(self, frame_id: int) -> bool:
        """判断是否应该更新记忆"""
        # 关键帧模式: 只在关键帧更新
        if self.use_keyframe and self.keyframe_indices:
            return frame_id in self.keyframe_indices
        
        # 定期更新
        return (frame_id - self.memory['last_update']) >= self.memory_update_interval
    
    def initialize(self, image, info: dict):
        """初始化跟踪器"""
        self.frame_id = 0
        H, W = image.shape[:2]
        
        self.state = list(info['init_bbox'])
        self.prev_image = image.copy()
        self.prev_bbox = list(info['init_bbox'])
        
        self.seq_name = info.get('seq_name', None)
        
        self.language_description = info.get('init_nlp', None)
        if not self.language_description:
            self.language_description = "the target object"
        
        # 生成初始记忆库
        print(f"[Memory] Generating initial memory...")
        self.memory = self._generate_memory(image, self.state)
        
        if self.debug >= 1:
            print(f"[Memory] Memory initialized:")
            print(f"  Appearance: {self.memory['appearance'][:100]}...")
            print(f"  Motion: {self.memory['motion'][:50]}...")
        
        # 关键帧跟踪
        if self.use_keyframe and self.keyframe_root:
            self.keyframe_indices = read_keyframe_indices(self.keyframe_root, self.seq_name)
            if self.keyframe_indices:
                print(f"[Memory] Keyframe mode: {len(self.keyframe_indices)} keyframes")
        
        # 可视化
        if self.debug >= 2:
            env = env_settings()
            self.vis_dir = os.path.join(env.results_path, 'qwen3vl_memory', 'vis', 
                                       self.seq_name or 'unknown')
            os.makedirs(self.vis_dir, exist_ok=True)
        
        if self.debug >= 1:
            mode_str = f"{self.mode} + memory-bank"
            if self.use_keyframe:
                mode_str += " + keyframe"
            print(f"[Memory] Initialize: bbox={self.state}, mode={mode_str}")
    
    def track(self, image, info: dict = None):
        """记忆库跟踪"""
        self.frame_id += 1
        H, W = image.shape[:2]
        
        # 稀疏跟踪: 非关键帧跳过
        if self.use_keyframe and self.keyframe_indices:
            if self.frame_id not in self.keyframe_indices:
                if self.debug >= 1:
                    print(f"[Memory] Frame {self.frame_id}: Skipped (non-keyframe)")
                return None
        
        try:
            # 准备图像
            prev_with_box = self._draw_bbox_on_image(
                self.prev_image, self.prev_bbox, color=(0, 0, 255), thickness=3
            )
            
            # 使用记忆库跟踪
            prompt = self._tracking_with_memory_prompt()
            output_text = self._run_inference([prev_with_box, image], prompt)
            
            if self.debug >= 1:
                kf_tag = "[KF]" if self.use_keyframe else ""
                print(f"[Memory] Frame {self.frame_id}{kf_tag}: {output_text[:100]}...")
            
            # 解析bbox
            bbox_xyxy = extract_bbox_from_model_output(output_text, W, H)
            
            if bbox_xyxy is not None:
                pred_bbox = xyxy_to_xywh(bbox_xyxy)
                self.state = pred_bbox
                
                # 可视化
                self._save_visualization(prev_with_box, image, pred_bbox, self.frame_id)
                
                # 更新上一帧
                self.prev_image = image.copy()
                self.prev_bbox = pred_bbox
                
                # 更新记忆库
                if self._should_update_memory(self.frame_id):
                    if self.debug >= 1:
                        print(f"[Memory] Updating memory at frame {self.frame_id}...")
                    self.memory = self._generate_memory(image, pred_bbox)
                
                # 缓存关键帧
                if self.use_keyframe:
                    self.keyframe_results[self.frame_id] = pred_bbox
            else:
                if self.debug >= 1:
                    print(f"[Memory] Frame {self.frame_id}: Failed to parse bbox")
                self.state = [0, 0, 0, 0]
                
        except Exception as e:
            print(f"[Memory] Error frame {self.frame_id}: {e}")
            import traceback
            traceback.print_exc()
            self.state = [0, 0, 0, 0]
        
        return {"target_bbox": self.state}


def get_tracker_class():
    return QWEN3VL_Memory

"""
Qwen3VL Hybrid Tracker V2: Three-Image + Memory Bank V2
终极方案V2: 三图 + 一次VLM调用同时输出bbox和状态

与V1的区别:
- V1: 需要额外VLM调用生成记忆
- V2: 跟踪时同时输出bbox和状态描述

跟踪范式:
- Image 1: 初始帧 + 初始框 (绿色) - 固定视觉锚点
- Image 2: 上一帧 + 预测框 (蓝色) - 短期运动
- Image 3: 当前帧 - 待预测
- Memory: 语义描述 - 文本锚点 (每帧从VLM输出更新)
"""
from lib.test.tracker.basetracker import BaseTracker
from lib.test.evaluation.environment import env_settings
import torch
import cv2
import numpy as np
import re
import json
import os
from PIL import Image
from typing import List, Dict

# 复用工具函数
from lib.test.tracker.qwen3vl import (
    extract_bbox_from_model_output,
    xyxy_to_xywh,
    numpy_to_base64,
    qwen3vl_api_chat,
    read_keyframe_indices
)


class QWEN3VL_Hybrid_V2(BaseTracker):
    """
    混合跟踪V2: 三图 + 记忆库 (一次VLM调用)
    
    最强配置:
    - 视觉锚点: 初始帧
    - 运动线索: 上一帧
    - 语义引导: 记忆库 (每帧更新)
    - 高效: 一次VLM调用同时输出bbox和state
    """
    
    def __init__(self, params, dataset_name):
        super(QWEN3VL_Hybrid_V2, self).__init__(params)
        self.params = params
        self.dataset_name = dataset_name
        
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
        print(f"[HybridV2] Loading: {model_name}")
        
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
        self.api_model = getattr(self.params, 'api_model', 'qwen3-vl-235b-a22b-instruct')
        self.api_base_url = getattr(self.params, 'api_base_url', 
                                     'https://dashscope.aliyuncs.com/compatible-mode/v1')
        self.api_key = getattr(self.params, 'api_key', os.environ.get('DASHSCOPE_API_KEY', ''))
        print(f"[HybridV2] API: {self.api_model}")
    
    def _draw_bbox(self, image: np.ndarray, bbox: List[float], color=(0,255,0), thickness=3) -> np.ndarray:
        """绘制bbox"""
        img = image.copy()
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)
        return img
    
    def _generate_initial_memory_prompt(self) -> str:
        """生成初始记忆prompt"""
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
    
    def _tracking_with_state_prompt(self) -> str:
        """
        三图+记忆库跟踪+状态输出的prompt (V2核心)
        输入: 3张图片 + 记忆库
        输出: bbox + state (一次VLM调用)
        """
        return (
            "# --- CORE TASK ---\n"
            "Track the target using semantic memory, initial appearance, and motion cues. Determine if target is visible and locate it.\n\n"
            
            "# --- SEMANTIC MEMORY ---\n"
            f"Appearance: {self.memory['appearance']}\n"
            f"Motion: {self.memory['motion']}\n"
            f"Context: {self.memory['context']}\n\n"
            
            "# --- VISUAL REFERENCE ---\n"
            "Image 1 (Initial - GREEN box): Ground truth target.\n"
            "Image 2 (Previous - BLUE box): Last prediction (may be inaccurate, use only for motion reference).\n"
            "Image 3 (Current): Find the target here.\n\n"
            
            "# --- OUTPUT REQUIREMENT ---\n"
            "Match the target based on: (1) Semantic memory, (2) Initial appearance (Image 1), (3) Motion from Image 2.\n"
            "Output JSON format with TWO fields:\n"
            "{\n"
            '  "bbox": [x1, y1, x2, y2],      // 0-1000 scale. Output [0,0,0,0] if target is invisible/occluded.\n'
            '  "evidence": "Describe matched features from memory, Image 1, and observed motion.",\n'
            '  "confidence": 0.95,            // Float between 0.0 (Lost) and 1.0 (Certain).\n'
            '  "state": {                     // Update current state for memory.\n'
            '    "appearance": "current appearance description",\n'
            '    "motion": "current motion state",\n'
            '    "context": "current context"\n'
            '  }\n'
            "}\n"
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
    
    def _parse_tracking_output(self, text: str, W: int, H: int):
        """
        解析VLM输出 (V2: 同时包含bbox和state)
        返回: (bbox_xywh, state_dict)
        """
        try:
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            
            data = json.loads(text.strip())
            
            # 解析bbox
            bbox_xyxy = None
            if 'bbox' in data:
                bbox_raw = data['bbox']
                if isinstance(bbox_raw, list) and len(bbox_raw) == 4:
                    bbox_xyxy = [float(v) for v in bbox_raw]
                    if max(bbox_xyxy) <= 1000:
                        bbox_xyxy = [bbox_xyxy[0]/1000*W, bbox_xyxy[1]/1000*H,
                                    bbox_xyxy[2]/1000*W, bbox_xyxy[3]/1000*H]
            
            if bbox_xyxy is None:
                bbox_xyxy = extract_bbox_from_model_output(text, W, H)
            
            # 解析state
            state = data.get('state', {})
            if isinstance(state, dict):
                def dict_to_str(val):
                    if isinstance(val, dict):
                        return ', '.join(f"{k}: {v}" for k, v in val.items())
                    return str(val)
                
                state = {
                    "appearance": dict_to_str(state.get("appearance", "")),
                    "motion": dict_to_str(state.get("motion", "")),
                    "context": dict_to_str(state.get("context", "")),
                    "last_update": self.frame_id
                }
            else:
                state = None
            
            return bbox_xyxy, state
            
        except Exception as e:
            if self.debug >= 1:
                print(f"[HybridV2] Parse error: {e}")
            bbox_xyxy = extract_bbox_from_model_output(text, W, H)
            return bbox_xyxy, None
    
    def _save_visualization(self, init_with_box: np.ndarray, prev_with_box: np.ndarray, 
                           search_img: np.ndarray, pred_bbox_xywh: List[float], frame_id: int):
        """保存混合V2可视化 (三帧+文本在下方padding)"""
        if self.debug < 2 or self.vis_dir is None:
            return
        
        result_img = self._draw_bbox(search_img, pred_bbox_xywh, (255,0,0), 3)
        
        # 调整三帧高度一致
        h1, w1 = init_with_box.shape[:2]
        h2, w2 = prev_with_box.shape[:2]
        h3, w3 = result_img.shape[:2]
        target_h = max(h1, h2, h3)
        
        def resize(img, h):
            return cv2.resize(img, (int(img.shape[1]*h/img.shape[0]), h)) if img.shape[0]!=h else img
        
        init_resized = resize(init_with_box, target_h)
        prev_resized = resize(prev_with_box, target_h)
        result_resized = resize(result_img, target_h)
        
        # 水平拼接三帧
        combined = np.hstack([init_resized, prev_resized, result_resized])
        
        # 在下方添加padding用于显示记忆文本
        padding_height = 120
        h_combined, w_combined = combined.shape[:2]
        padding = np.ones((padding_height, w_combined, 3), dtype=np.uint8) * 255
        
        # 在padding上写记忆文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 25
        cv2.putText(padding, f"Frame {frame_id} - Hybrid V2 (3-Img + Memory):", (10,y), font, 0.6, (0,0,255), 2)
        y += 30
        cv2.putText(padding, f"App: {self.memory.get('appearance','')[:100]}", (10,y), font, 0.4, (0,0,0), 1)
        y += 25
        cv2.putText(padding, f"Motion: {self.memory.get('motion','')[:80]}", (10,y), font, 0.4, (0,0,0), 1)
        y += 25
        cv2.putText(padding, f"Context: {self.memory.get('context','')[:80]}", (10,y), font, 0.4, (0,0,0), 1)
        
        # 垂直拼接: 图片 + padding
        final_img = np.vstack([combined, padding])
        
        # 添加图片标注
        cv2.putText(final_img, "Init (Green)", (10, target_h-10), font, 0.6, (0,255,0), 2)
        cv2.putText(final_img, "Prev (Blue)", (init_resized.shape[1]+10, target_h-10), font, 0.6, (255,0,0), 2)
        cv2.putText(final_img, "Current (Red)", (init_resized.shape[1]+prev_resized.shape[1]+10, target_h-10), font, 0.6, (0,0,255), 2)
        
        # 保存
        vis_path = os.path.join(self.vis_dir, f"{self.seq_name}_{frame_id:04d}.jpg")
        cv2.imwrite(vis_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
        
        if self.debug >= 3:
            cv2.imshow('Hybrid-V2', cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
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
        print(f"[HybridV2] Generating initial memory...")
        img_with_box = self._draw_bbox(image, self.state, (0,255,0))
        prompt = self._generate_initial_memory_prompt()
        output = self._run_inference([img_with_box], prompt)
        
        try:
            output = re.sub(r'```json\s*', '', output)
            output = re.sub(r'```\s*', '', output)
            data = json.loads(output.strip())
            
            def dict_to_str(val):
                if isinstance(val, dict):
                    return ', '.join(f"{k}: {v}" for k, v in val.items())
                return str(val)
            
            self.memory = {
                "appearance": dict_to_str(data.get("appearance", "")),
                "motion": dict_to_str(data.get("motion", "")),
                "context": dict_to_str(data.get("context", "")),
                "last_update": 0
            }
        except:
            self.memory = {
                "appearance": self.language_description,
                "motion": "unknown",
                "context": "unknown",
                "last_update": 0
            }
        
        if self.debug >= 1:
            print(f"[HybridV2] Initial memory: {self.memory['appearance'][:80]}...")
        
        # 关键帧
        if self.use_keyframe and self.keyframe_root:
            self.keyframe_indices = read_keyframe_indices(self.keyframe_root, self.seq_name)
            if self.keyframe_indices:
                print(f"[HybridV2] Keyframes: {len(self.keyframe_indices)}")
        
        # 可视化
        if self.debug >= 2:
            env = env_settings()
            self.vis_dir = os.path.join(env.results_path, 'qwen3vl_hybrid_v2', 'vis', 
                                       self.seq_name or 'unknown')
            os.makedirs(self.vis_dir, exist_ok=True)
        
        mode_str = f"{self.mode} + three-image + memory-v2"
        if self.use_keyframe:
            mode_str += " + keyframe"
        print(f"[HybridV2] Init: {mode_str}")
    
    def track(self, image, info: dict = None):
        """混合跟踪V2 (一次调用同时获取bbox和state)"""
        self.frame_id += 1
        H, W = image.shape[:2]
        
        # 稀疏跟踪
        if self.use_keyframe and self.keyframe_indices:
            if self.frame_id not in self.keyframe_indices:
                if self.debug >= 1:
                    print(f"[HybridV2] Frame {self.frame_id}: Skipped")
                return None
        
        try:
            # 准备三图
            init_with_box = self._draw_bbox(self.init_image, self.init_bbox, (0,255,0))
            prev_with_box = self._draw_bbox(self.prev_image, self.prev_bbox, (0,0,255))
            
            # V2核心: 一次VLM调用同时获取bbox和state
            prompt = self._tracking_with_state_prompt()
            output = self._run_inference([init_with_box, prev_with_box, image], prompt)
            
            if self.debug >= 1:
                kf = "[KF]" if self.use_keyframe else ""
                print(f"[HybridV2] Frame {self.frame_id}{kf}: {output[:80]}...")
            
            # 解析bbox和state
            bbox_xyxy, new_state = self._parse_tracking_output(output, W, H)
            
            if bbox_xyxy:
                pred_bbox = xyxy_to_xywh(bbox_xyxy)
                self.state = pred_bbox
                
                # 可视化 (三帧)
                self._save_visualization(init_with_box, prev_with_box, image, pred_bbox, self.frame_id)
                
                # V2核心: 如果VLM输出了state,直接更新记忆库
                if new_state is not None:
                    self.memory = new_state
                    if self.debug >= 1:
                        print(f"[HybridV2] Memory updated from VLM output")
                
                # 更新上一帧
                self.prev_image = image.copy()
                self.prev_bbox = pred_bbox
                
                # 缓存
                if self.use_keyframe:
                    self.keyframe_results[self.frame_id] = pred_bbox
            else:
                if self.debug >= 1:
                    print(f"[HybridV2] Frame {self.frame_id}: Parse failed")
                self.state = [0,0,0,0]
        
        except Exception as e:
            print(f"[HybridV2] Error {self.frame_id}: {e}")
            import traceback
            traceback.print_exc()
            self.state = [0,0,0,0]
        
        return {"target_bbox": self.state}


def get_tracker_class():
    return QWEN3VL_Hybrid_V2

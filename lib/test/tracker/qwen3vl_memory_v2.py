"""
Qwen3VL Memory Bank Tracker V2
改进方案: 一次VLM调用同时输出bbox和状态描述,更新记忆库

与V1的区别:
- V1: 需要额外VLM调用生成记忆 (成本高)
- V2: 跟踪时同时输出bbox和状态描述 (成本低,效率高)

跟踪范式:
- 初始化: 生成初始记忆
- 跟踪: 一次VLM调用输出 {"bbox": [...], "state": {...}}
- 更新: 直接使用VLM输出的state更新记忆库
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


class QWEN3VL_Memory_V2(BaseTracker):
    """
    记忆库跟踪V2: 一次VLM调用同时输出bbox和状态
    
    改进:
    - 减少VLM调用次数
    - 实时状态描述
    - 更高效的记忆更新
    """
    
    def __init__(self, params, dataset_name):
        super(QWEN3VL_Memory_V2, self).__init__(params)
        self.params = params
        self.dataset_name = dataset_name
        
        self.mode = getattr(params, 'mode', 'local')
        
        if self.mode == 'local':
            self._load_local_model()
        else:
            self._setup_api()
        
        self.state = None
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
        from transformers import AutoProcessor
        
        model_name = getattr(self.params, 'model_name', 'Qwen/Qwen3-VL-4B-Instruct')
        print(f"[MemoryV2] Loading: {model_name}")
        
        if 'qwen3' in model_name.lower():
            from transformers import Qwen3VLForConditionalGeneration
            model_class = Qwen3VLForConditionalGeneration
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model_class = Qwen2_5_VLForConditionalGeneration
        
        try:
            self.model = model_class.from_pretrained(
                model_name, torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2", device_map="auto"
            )
        except:
            self.model = model_class.from_pretrained(
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
        print(f"[MemoryV2] API: {self.api_model}")
    
    def _draw_bbox(self, image: np.ndarray, bbox: List[float], color=(0,255,0), thickness=3) -> np.ndarray:
        """绘制bbox"""
        img = image.copy()
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)
        return img
    
    def _generate_initial_memory_prompt(self) -> str:
        """
        生成初始记忆的prompt (只在初始化时调用一次)
        """
        return (
            "Analyze the target object marked by the green bounding box. "
            "Provide a detailed description in JSON format: "
            '{"appearance": "color, shape, texture, features", '
            '"motion": "current motion state", '
            '"context": "surrounding objects and position"}. '
            "Be specific. Output ONLY the JSON object."
        )
    
    def _tracking_with_state_prompt(self) -> str:
        """
        跟踪+状态描述的prompt (V2核心改进)
        一次VLM调用同时输出bbox和当前状态
        """
        return (
            # 提供记忆库作为参考
            f"Target memory: appearance is {self.memory['appearance']}, "
            f"motion is {self.memory['motion']}, "
            f"context is {self.memory['context']}. "
            # 图像说明
            f"The first image shows the previous frame with the predicted target location marked by a blue bounding box (may not be accurate, use only for motion reference). "
            f"The second image is the current frame. "
            # 任务: 同时输出bbox和状态描述
            f"Locate the target that matches the memory description in the second image. "
            f"Output JSON format with TWO fields: "
            f'"bbox": [x1, y1, x2, y2] in 0-1000 scale, '
            f'"state": {{"appearance": "current appearance", "motion": "current motion", "context": "current context"}}.'
        )
    
    def _run_inference(self, images: List[np.ndarray], prompt: str) -> str:
        """运行推理"""
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
        解析VLM输出 (V2核心: 同时包含bbox和state)
        返回: (bbox_xywh, state_dict)
        """
        try:
            # 清理文本
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            
            data = json.loads(text.strip())
            
            # 解析bbox
            bbox_xyxy = None
            if 'bbox' in data:
                bbox_raw = data['bbox']
                if isinstance(bbox_raw, list) and len(bbox_raw) == 4:
                    bbox_xyxy = [float(v) for v in bbox_raw]
                    # 转换坐标
                    if max(bbox_xyxy) <= 1000:
                        bbox_xyxy = [bbox_xyxy[0]/1000*W, bbox_xyxy[1]/1000*H,
                                    bbox_xyxy[2]/1000*W, bbox_xyxy[3]/1000*H]
            
            if bbox_xyxy is None:
                # 尝试其他字段
                bbox_xyxy = extract_bbox_from_model_output(text, W, H)
            
            # 解析state
            state = data.get('state', {})
            if isinstance(state, dict):
                # 将字典值转为字符串
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
                print(f"[MemoryV2] Parse error: {e}")
            # 尝试只解析bbox
            bbox_xyxy = extract_bbox_from_model_output(text, W, H)
            return bbox_xyxy, None
    
    def _save_visualization(self, prev_with_box: np.ndarray, search_img: np.ndarray, 
                           pred_bbox_xywh: List[float], frame_id: int):
        """保存记忆库V2可视化 (文本在下方padding)"""
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
        combined = np.hstack([prev_resized, result_resized])
        padding_height = 120
        h_combined, w_combined = combined.shape[:2]
        padding = np.ones((padding_height, w_combined, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 25
        cv2.putText(padding, f"Frame {frame_id} - Memory V2:", (10,y), font, 0.6, (0,0,255), 2)
        y += 30
        cv2.putText(padding, f"App: {self.memory.get('appearance','')[:100]}", (10,y), font, 0.4, (0,0,0), 1)
        y += 25
        cv2.putText(padding, f"Motion: {self.memory.get('motion','')[:80]}", (10,y), font, 0.4, (0,0,0), 1)
        y += 25
        cv2.putText(padding, f"Context: {self.memory.get('context','')[:80]}", (10,y), font, 0.4, (0,0,0), 1)
        final_img = np.vstack([combined, padding])
        cv2.putText(final_img, "Prev (Blue)", (10, target_h-10), font, 0.6, (255,0,0), 2)
        cv2.putText(final_img, "Current (Red)", (prev_resized.shape[1]+10, target_h-10), font, 0.6, (0,0,255), 2)
        vis_path = os.path.join(self.vis_dir, f"{self.seq_name}_{frame_id:04d}.jpg")
        cv2.imwrite(vis_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
        if self.debug >= 3:
            cv2.imshow('Memory-V2', cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
    
    def initialize(self, image, info: dict):
        """初始化"""
        self.frame_id = 0
        self.state = list(info['init_bbox'])
        self.prev_image = image.copy()
        self.prev_bbox = list(info['init_bbox'])
        
        self.seq_name = info.get('seq_name', None)
        self.language_description = info.get('init_nlp', None) or "the target object"
        
        # 生成初始记忆 (只调用一次)
        print(f"[MemoryV2] Generating initial memory...")
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
            print(f"[MemoryV2] Initial memory: {self.memory['appearance'][:80]}...")
        
        # 关键帧
        if self.use_keyframe and self.keyframe_root:
            self.keyframe_indices = read_keyframe_indices(self.keyframe_root, self.seq_name)
            if self.keyframe_indices:
                print(f"[MemoryV2] Keyframes: {len(self.keyframe_indices)}")
        
        # 可视化
        if self.debug >= 2:
            env = env_settings()
            self.vis_dir = os.path.join(env.results_path, 'qwen3vl_memory_v2', 'vis', 
                                       self.seq_name or 'unknown')
            os.makedirs(self.vis_dir, exist_ok=True)
    
    def track(self, image, info: dict = None):
        """跟踪 (V2: 一次调用同时获取bbox和state)"""
        self.frame_id += 1
        H, W = image.shape[:2]
        
        # 稀疏跟踪
        if self.use_keyframe and self.keyframe_indices:
            if self.frame_id not in self.keyframe_indices:
                if self.debug >= 1:
                    print(f"[MemoryV2] Frame {self.frame_id}: Skipped")
                return None
        
        try:
            # 准备图像
            prev_with_box = self._draw_bbox(self.prev_image, self.prev_bbox, (0,0,255), 3)
            
            # V2核心: 一次VLM调用同时获取bbox和state
            prompt = self._tracking_with_state_prompt()
            output = self._run_inference([prev_with_box, image], prompt)
            
            if self.debug >= 1:
                kf = "[KF]" if self.use_keyframe else ""
                print(f"[MemoryV2] Frame {self.frame_id}{kf}: {output[:100]}...")
            
            # 解析bbox和state
            bbox_xyxy, new_state = self._parse_tracking_output(output, W, H)
            
            if bbox_xyxy is not None:
                pred_bbox = xyxy_to_xywh(bbox_xyxy)
                self.state = pred_bbox
                
                # 可视化
                self._save_visualization(prev_with_box, image, pred_bbox, self.frame_id)
                
                # V2核心: 如果VLM输出了state,直接更新记忆库
                if new_state is not None:
                    self.memory = new_state
                    if self.debug >= 1:
                        print(f"[MemoryV2] Memory updated from VLM output")
                
                # 更新上一帧
                self.prev_image = image.copy()
                self.prev_bbox = pred_bbox
                
                if self.use_keyframe:
                    self.keyframe_results[self.frame_id] = pred_bbox
            else:
                if self.debug >= 1:
                    print(f"[MemoryV2] Frame {self.frame_id}: Parse failed")
                self.state = [0,0,0,0]
        
        except Exception as e:
            print(f"[MemoryV2] Error {self.frame_id}: {e}")
            import traceback
            traceback.print_exc()
            self.state = [0,0,0,0]
        
        return {"target_bbox": self.state}


def get_tracker_class():
    return QWEN3VL_Memory_V2

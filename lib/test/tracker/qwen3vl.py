"""
Qwen3VL Tracker: VLM-based tracker using Qwen3-VL
支持:
1. 本地推理 (Local) - 加载本地模型
2. API推理 (API) - 调用OpenAI兼容API (支持多线程!)

跟踪范式:
- 模板帧上画框,不在prompt中写坐标
- 初始化: 第一帧 + 目标框 + 语言描述  
- Track: 模板帧(带框) + 搜索帧 → VLM预测bbox
- 动态更新模板 (当前帧 + 预测框)
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
import io
from PIL import Image
from typing import List, Optional, Tuple


# ============== Bbox Parsing Utils ==============

def _strip_code_fence(text: str) -> str:
    """去掉 ```json / ``` 等包裹"""
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _safe_float_list(x):
    """尝试把输入转换成长度为4的float list"""
    if isinstance(x, (list, tuple)) and len(x) == 4:
        try:
            return [float(v) for v in x]
        except Exception:
            return None
    return None


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _fix_and_clip_bbox(b: List[float], w: int, h: int) -> List[float]:
    """修正顺序并裁剪到图像范围"""
    x1, y1, x2, y2 = b
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x1 = _clamp(x1, 0.0, float(w - 1))
    y1 = _clamp(y1, 0.0, float(h - 1))
    x2 = _clamp(x2, 0.0, float(w - 1))
    y2 = _clamp(y2, 0.0, float(h - 1))

    if abs(x2 - x1) < 1.0:
        x2 = _clamp(x1 + 1.0, 0.0, float(w - 1))
    if abs(y2 - y1) < 1.0:
        y2 = _clamp(y1 + 1.0, 0.0, float(h - 1))

    return [x1, y1, x2, y2]


def _convert_to_pixel_bbox(b: List[float], w: int, h: int) -> List[float]:
    """支持 0-1 / 0-1000 / 像素坐标"""
    x1, y1, x2, y2 = b
    maxv = max(x1, y1, x2, y2)
    minv = min(x1, y1, x2, y2)

    if 0.0 <= minv and maxv <= 1.0:
        return [x1 * w, y1 * h, x2 * w, y2 * h]

    if 0.0 <= minv and maxv <= 1000.0:
        return [(x1 / 1000.0) * w, (y1 / 1000.0) * h, (x2 / 1000.0) * w, (y2 / 1000.0) * h]

    return [x1, y1, x2, y2]


def extract_bbox_from_model_output(text: str, img_width: int, img_height: int) -> Optional[List[float]]:
    """从模型输出解析bbox,返回[x1,y1,x2,y2]像素坐标"""
    raw = text or ""
    t = _strip_code_fence(raw)
    if not t:
        return None

    # 尝试JSON解析
    try:
        data = json.loads(t)
        if isinstance(data, dict) and "bbox_2d" in data:
            b = _safe_float_list(data["bbox_2d"])
            if b:
                b = _convert_to_pixel_bbox(b, img_width, img_height)
                return _fix_and_clip_bbox(b, img_width, img_height)

        elif isinstance(data, list):
            if len(data) == 4 and all(isinstance(x, (int, float)) for x in data):
                b = [float(x) for x in data]
                b = _convert_to_pixel_bbox(b, img_width, img_height)
                return _fix_and_clip_bbox(b, img_width, img_height)

    except Exception:
        pass

    # 正则匹配 [x1,y1,x2,y2]
    matches = re.findall(r"\[([^\[\]]+)\]", t)
    for m in matches:
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m)
        if len(nums) == 4:
            try:
                b = [float(x) for x in nums]
                b = _convert_to_pixel_bbox(b, img_width, img_height)
                return _fix_and_clip_bbox(b, img_width, img_height)
            except Exception:
                continue

    return None


def xyxy_to_xywh(bbox_xyxy: List[float]) -> List[float]:
    x1, y1, x2, y2 = bbox_xyxy
    return [x1, y1, x2 - x1, y2 - y1]


def xywh_to_xyxy(bbox_xywh: List[float]) -> List[float]:
    x, y, w, h = bbox_xywh
    return [x, y, x + w, y + h]


def numpy_to_base64(image: np.ndarray) -> str:
    """将numpy图像转换为base64字符串 (不存本地文件)"""
    # RGB to BGR
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image
    
    # 编码为JPEG
    _, buffer = cv2.imencode('.jpg', image_bgr)
    b64 = base64.b64encode(buffer).decode('utf-8')
    return b64


# ============== API Inference ==============

def qwen3vl_api_chat(
    images_b64: List[str],
    prompt: str,
    model_name: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.1,
    max_tokens: int = 256,
    retries: int = 3,
) -> str:
    """
    OpenAI 兼容 API 推理 (使用base64图像,不存文件)
    """
    from openai import OpenAI
    
    content = []
    for b64 in images_b64:
        content.append({
            "type": "image_url", 
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
    content.append({"type": "text", "text": prompt})
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [{"role": "user", "content": content}]
    
    last_err = None
    for _ in range(max(1, retries)):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    
    raise RuntimeError(f"API调用失败: {last_err}")


# ============== Qwen3VL Tracker ==============

class QWEN3VL(BaseTracker):
    """
    VLM-based tracker using Qwen3-VL
    
    模式:
    - mode='local': 本地推理
    - mode='api': API推理 (支持多线程)
    
    debug:
    - 0: 无输出
    - 1: 打印关键信息
    - 2: 保存可视化图片到results目录
    - 3: 保存图片 + 实时显示窗口
    """
    
    def __init__(self, params, dataset_name):
        super(QWEN3VL, self).__init__(params)
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
        self.template_image = None
        self.template_bbox = None
        self.language_description = None
        self.frame_id = 0
        self.seq_name = None
        
        # Debug & Visualization
        self.debug = getattr(params, 'debug', 0)
        self.vis_dir = None  # 在initialize时根据results路径设置
    
    def _load_local_model(self):
        """加载本地模型"""
        from transformers import AutoProcessor
        
        model_path = getattr(self.params, 'model_path', None)
        model_name = getattr(self.params, 'model_name', 'Qwen/Qwen3-VL-4B-Instruct')
        
        actual_path = model_path or model_name
        print(f"[Qwen3VL] Loading local model: {actual_path}")
        
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
        print(f"[Qwen3VL] Model loaded: {model_class.__name__}")
    
    def _setup_api(self):
        """配置API"""
        self.api_model = getattr(self.params, 'api_model', 'qwen3-vl-235b-a22b-instruct')
        self.api_base_url = getattr(self.params, 'api_base_url', 
                                     'https://dashscope.aliyuncs.com/compatible-mode/v1')
        self.api_key = getattr(self.params, 'api_key', os.environ.get('DASHSCOPE_API_KEY', ''))
        
        print(f"[Qwen3VL] API mode: {self.api_model}")
    
    def _draw_bbox_on_image(self, image: np.ndarray, bbox_xywh: List[float], 
                            color=(0, 255, 0), thickness=3) -> np.ndarray:
        """在图像上绘制bbox"""
        img = image.copy()
        x, y, w, h = [int(v) for v in bbox_xywh]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        return img
    
    def _build_tracking_prompt(self, description: str) -> str:
        """构建跟踪prompt"""
        return (
            f"The first image shows the template with the target object marked by a green bounding box. "
            f"The target is: {description}. "
            f"The second image is the current frame. "
            f"Locate the same target in the second image, output its bbox coordinates using JSON format."
        )
    
    def _run_inference(self, template_img: np.ndarray, search_img: np.ndarray, prompt: str) -> str:
        """运行推理 (图像在内存中,不存文件)"""
        if self.mode == 'api':
            # API模式: 转base64
            template_b64 = numpy_to_base64(template_img)
            search_b64 = numpy_to_base64(search_img)
            return qwen3vl_api_chat(
                images_b64=[template_b64, search_b64],
                prompt=prompt,
                model_name=self.api_model,
                base_url=self.api_base_url,
                api_key=self.api_key,
            )
        else:
            return self._run_local_inference(template_img, search_img, prompt)
    
    def _run_local_inference(self, template_img: np.ndarray, search_img: np.ndarray, prompt: str) -> str:
        """本地推理 (使用PIL Image,不存文件)"""
        # numpy RGB to PIL
        template_pil = Image.fromarray(template_img)
        search_pil = Image.fromarray(search_img)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": template_pil},
                    {"type": "image", "image": search_pil},
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
    
    def _save_visualization(self, template_with_box: np.ndarray, 
                           search_img: np.ndarray, 
                           pred_bbox_xywh: List[float],
                           frame_id: int):
        """保存可视化到results目录"""
        if self.debug < 2 or self.vis_dir is None:
            return
        
        result_img = self._draw_bbox_on_image(search_img, pred_bbox_xywh, 
                                              color=(0, 0, 255), thickness=3)
        
        h1, w1 = template_with_box.shape[:2]
        h2, w2 = result_img.shape[:2]
        
        if h1 != h2:
            scale = h1 / h2
            new_w2 = int(w2 * scale)
            result_img = cv2.resize(result_img, (new_w2, h1))
        
        combined = np.hstack([template_with_box, result_img])
        cv2.putText(combined, f"Frame {frame_id}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        seq_name = self.seq_name or "unknown"
        vis_path = os.path.join(self.vis_dir, f"{seq_name}_{frame_id:04d}.jpg")
        cv2.imwrite(vis_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        
        if self.debug >= 3:
            cv2.imshow('Qwen3VL Tracking', cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
    
    def initialize(self, image, info: dict):
        """初始化跟踪器"""
        self.frame_id = 0
        H, W = image.shape[:2]
        
        self.state = list(info['init_bbox'])
        self.template_image = image.copy()
        self.template_bbox = list(info['init_bbox'])
        self.img_height, self.img_width = H, W
        self.seq_name = info.get('seq_name', None)
        
        self.language_description = info.get('init_nlp', None)
        if not self.language_description:
            self.language_description = "the target object marked in green box"
        
        # 设置可视化目录 (与results路径一致)
        if self.debug >= 2:
            env = env_settings()
            self.vis_dir = os.path.join(env.results_path, 'qwen3vl', 'vis', self.seq_name or 'unknown')
            os.makedirs(self.vis_dir, exist_ok=True)
        
        if self.debug >= 1:
            print(f"[Qwen3VL] Initialize: bbox={self.state}, mode={self.mode}")
    
    def track(self, image, info: dict = None):
        """跟踪当前帧"""
        self.frame_id += 1
        H, W = image.shape[:2]
        
        try:
            # 1. 模板画框 (在内存中)
            template_with_bbox = self._draw_bbox_on_image(
                self.template_image, self.template_bbox, color=(0, 255, 0), thickness=3
            )
            
            # 2. 构建prompt
            prompt = self._build_tracking_prompt(self.language_description)
            
            # 3. 推理 (图像在内存中,不存文件)
            output_text = self._run_inference(template_with_bbox, image, prompt)
            
            if self.debug >= 1:
                print(f"[Qwen3VL] Frame {self.frame_id}: {output_text[:100]}...")
            
            # 4. 解析bbox
            bbox_xyxy = extract_bbox_from_model_output(output_text, W, H)
            
            if bbox_xyxy is not None:
                pred_bbox = xyxy_to_xywh(bbox_xyxy)
                self.state = pred_bbox
                
                self._save_visualization(template_with_bbox, image, pred_bbox, self.frame_id)
                
                # 更新模板
                self.template_image = image.copy()
                self.template_bbox = pred_bbox
            else:
                # 解析失败,返回[0,0,0,0]
                if self.debug >= 1:
                    print(f"[Qwen3VL] Frame {self.frame_id}: Failed to parse bbox, return [0,0,0,0]")
                self.state = [0, 0, 0, 0]
                
        except Exception as e:
            print(f"[Qwen3VL] Error frame {self.frame_id}: {e}")
            self.state = [0, 0, 0, 0]
        
        return {"target_bbox": self.state}


def get_tracker_class():
    return QWEN3VL

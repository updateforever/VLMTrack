"""
VLM Tracker通用工具函数模块

包含所有tracker共用的工具函数：
- Bbox解析和转换
- 图像处理
- API调用
"""
import re
import json
import time
import base64
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional


# ============== Bbox解析和转换 ==============

def strip_code_fence(text: str) -> str:
    """去掉 ```json / ``` 等包裹"""
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def safe_float_list(x):
    """尝试把输入转换成长度为4的float list"""
    if isinstance(x, (list, tuple)) and len(x) == 4:
        try:
            return [float(v) for v in x]
        except Exception:
            return None
    return None


def clamp(v, lo, hi):
    """限制值在范围内"""
    return max(lo, min(hi, v))


def fix_and_clip_bbox(b: List[float], w: int, h: int) -> List[float]:
    """修正bbox顺序并裁剪到图像范围"""
    x1, y1, x2, y2 = b
    
    # 确保x1 < x2, y1 < y2
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    
    # 裁剪到图像范围
    x1 = clamp(x1, 0.0, float(w - 1))
    y1 = clamp(y1, 0.0, float(h - 1))
    x2 = clamp(x2, 0.0, float(w - 1))
    y2 = clamp(y2, 0.0, float(h - 1))
    
    # 确保最小宽高
    if abs(x2 - x1) < 1.0:
        x2 = clamp(x1 + 1.0, 0.0, float(w - 1))
    if abs(y2 - y1) < 1.0:
        y2 = clamp(y1 + 1.0, 0.0, float(h - 1))
    
    return [x1, y1, x2, y2]


def convert_to_pixel_bbox(b: List[float], w: int, h: int) -> List[float]:
    """
    将bbox转换为像素坐标
    支持三种输入格式:
    - 0-1归一化坐标
    - 0-1000坐标
    - 像素坐标
    """
    x1, y1, x2, y2 = b
    maxv = max(x1, y1, x2, y2)
    minv = min(x1, y1, x2, y2)
    
    # 0-1归一化
    if 0.0 <= minv and maxv <= 1.0:
        return [x1 * w, y1 * h, x2 * w, y2 * h]
    
    # 0-1000坐标
    if 0.0 <= minv and maxv <= 1000.0:
        return [(x1 / 1000.0) * w, (y1 / 1000.0) * h, 
                (x2 / 1000.0) * w, (y2 / 1000.0) * h]
    
    # 已经是像素坐标
    return [x1, y1, x2, y2]


def parse_bbox_from_text(text: str, img_width: int, img_height: int) -> Optional[List[float]]:
    """
    从VLM输出文本解析bbox
    
    Returns:
        [x1, y1, x2, y2] 像素坐标，失败返回None
    """
    raw = text or ""
    t = strip_code_fence(raw)
    if not t:
        return None
    
    # 尝试JSON解析
    try:
        data = json.loads(t)
        
        # 格式1: {"bbox": [x1, y1, x2, y2]}
        if isinstance(data, dict) and "bbox" in data:
            b = safe_float_list(data["bbox"])
            if b:
                b = convert_to_pixel_bbox(b, img_width, img_height)
                return fix_and_clip_bbox(b, img_width, img_height)
        
        # 格式2: {"bbox_2d": [x1, y1, x2, y2]}
        if isinstance(data, dict) and "bbox_2d" in data:
            b = safe_float_list(data["bbox_2d"])
            if b:
                b = convert_to_pixel_bbox(b, img_width, img_height)
                return fix_and_clip_bbox(b, img_width, img_height)
        
        # 格式3: [x1, y1, x2, y2]
        elif isinstance(data, list):
            if len(data) == 4 and all(isinstance(x, (int, float)) for x in data):
                b = [float(x) for x in data]
                b = convert_to_pixel_bbox(b, img_width, img_height)
                return fix_and_clip_bbox(b, img_width, img_height)
    
    except Exception:
        pass
    
    # 正则匹配 [x1,y1,x2,y2]
    matches = re.findall(r"\[([^\[\]]+)\]", t)
    for m in matches:
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m)
        if len(nums) == 4:
            try:
                b = [float(x) for x in nums]
                b = convert_to_pixel_bbox(b, img_width, img_height)
                return fix_and_clip_bbox(b, img_width, img_height)
            except Exception:
                continue
    
    return None


def xyxy_to_xywh(bbox_xyxy: List[float]) -> List[float]:
    """XYXY格式转XYWH格式"""
    x1, y1, x2, y2 = bbox_xyxy
    return [x1, y1, x2 - x1, y2 - y1]


def xywh_to_xyxy(bbox_xywh: List[float]) -> List[float]:
    """XYWH格式转XYXY格式"""
    x, y, w, h = bbox_xywh
    return [x, y, x + w, y + h]


# ============== 图像处理 ==============

def draw_bbox(image: np.ndarray, bbox_xywh: List[float], 
              color=(0, 255, 0), thickness=3) -> np.ndarray:
    """在图像上绘制bbox"""
    img = image.copy()
    x, y, w, h = [int(v) for v in bbox_xywh]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    return img


def numpy_to_base64(image: np.ndarray) -> str:
    """将numpy图像转换为base64字符串"""
    # RGB to BGR
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image
    
    # 编码为JPEG
    _, buffer = cv2.imencode('.jpg', image_bgr)
    b64 = base64.b64encode(buffer).decode('utf-8')
    return b64


# ============== API调用 ==============

def call_vlm_api(
    images_b64: List[str],
    prompt: str,
    model_name: str,
    base_url: str = '',
    api_key: str = '',
    client=None,
    temperature: float = 0.1,
    max_tokens: int = 512,
    retries: int = 3,
) -> str:
    """
    调用OpenAI兼容的VLM API。

    Args:
        images_b64: base64编码的图像列表
        prompt:     文本prompt
        model_name: 模型名称
        base_url:   API基础URL（client无效时使用）
        api_key:    API密鄉（client无效时使用）
        client:     预建的OpenAI客户端，传入可避免重复创建
        temperature: 温度参数
        max_tokens: 最大token数
        retries:    重试次数

    Returns:
        模型输出文本
    """
    from openai import OpenAI

    _client = client or OpenAI(api_key=api_key, base_url=base_url)

    # 构建消息内容
    content = []
    for b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    last_err = None
    for _ in range(max(1, retries)):
        try:
            resp = _client.chat.completions.create(
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


# ============== 记忆库工具 ==============

def dict_to_str(val):
    """将字典或其他值转为字符串"""
    if isinstance(val, dict):
        return ', '.join(f"{k}: {v}" for k, v in val.items())
    return str(val)


def parse_memory_state(text: str) -> Optional[dict]:
    """
    从VLM输出解析记忆状态

    Returns:
        {"appearance": str, "motion": str, "context": str} 或 None
    """
    try:
        t = strip_code_fence(text)
        data = json.loads(t)

        if isinstance(data, dict) and "state" in data:
            state = data["state"]
            if isinstance(state, dict):
                return {
                    "appearance": dict_to_str(state.get("appearance", "")),
                    "motion": dict_to_str(state.get("motion", "")),
                    "context": dict_to_str(state.get("context", ""))
                }

        # 直接是状态字典
        if isinstance(data, dict) and "appearance" in data:
            return {
                "appearance": dict_to_str(data.get("appearance", "")),
                "motion": dict_to_str(data.get("motion", "")),
                "context": dict_to_str(data.get("context", ""))
            }

    except Exception:
        pass

    return None


def parse_init_cognition(text: str) -> Optional[str]:
    """
    解析 Mosaic 初始化认知结果。

    支持格式:
      {"init_cognition": "..."}
      {"init_story": "..."}
      {"story": "..."}
    """
    try:
        t = strip_code_fence(text)
        data = json.loads(t)
        if isinstance(data, dict):
            cognition = (
                data.get("init_cognition", "")
                or data.get("init_story", "")
                or data.get("story", "")
            )
            if isinstance(cognition, str):
                cognition = cognition.strip()
                if cognition:
                    return cognition
    except Exception:
        pass
    return None


def parse_init_story(text: str) -> Optional[str]:
    """兼容旧调用，等价于 parse_init_cognition。"""
    return parse_init_cognition(text)


# ============== 认知跟踪输出解析 ==============

def parse_cognitive_output(text: str, img_width: int, img_height: int) -> Optional[dict]:
    """
    解析认知跟踪输出（新版结构化格式）

    Returns:
        {
            'target_status': str,
            'environment_status': List[str],
            'bbox': List[float],  # XYWH format
            'tracking_evidence': str,
            'confidence': float
        } 或 None
    """
    try:
        t = strip_code_fence(text)
        data = json.loads(t)

        # 验证 target_status
        valid_target_status = [
            'normal', 'partially_occluded', 'fully_occluded',
            'out_of_view', 'disappeared', 'reappeared'
        ]
        target_status = data.get('target_status', 'normal')
        if target_status not in valid_target_status:
            target_status = 'normal'

        # 验证 environment_status
        valid_env_status = [
            'normal', 'low_light', 'high_light', 'motion_blur',
            'scene_change', 'viewpoint_change', 'scale_change',
            'crowded', 'background_clutter'
        ]
        env_status = data.get('environment_status', ['normal'])
        if not isinstance(env_status, list):
            env_status = ['normal']
        env_status = [s for s in env_status if s in valid_env_status]
        if not env_status:
            env_status = ['normal']

        # 解析 bbox
        bbox_xyxy = None
        if 'bbox' in data:
            b = safe_float_list(data['bbox'])
            if b and b != [0, 0, 0, 0]:
                b = convert_to_pixel_bbox(b, img_width, img_height)
                bbox_xyxy = fix_and_clip_bbox(b, img_width, img_height)

        # 验证 bbox 与 target_status 的一致性
        visible_status = ['normal', 'partially_occluded', 'reappeared']
        if target_status in visible_status and bbox_xyxy is None:
            # 状态说可见但没有 bbox，标记为解析失败
            return None

        if target_status not in visible_status and bbox_xyxy is not None:
            # 状态说不可见但有 bbox，强制清零
            bbox_xyxy = None

        bbox_xywh = xyxy_to_xywh(bbox_xyxy) if bbox_xyxy else [0, 0, 0, 0]

        return {
            'target_status': target_status,
            'environment_status': env_status,
            'bbox': bbox_xywh,
            'tracking_evidence': data.get('tracking_evidence', ''),
            'confidence': float(data.get('confidence', 0.5))
        }

    except Exception:
        pass

    return None


def parse_mosaic_output(text: str, img_width: int, img_height: int) -> Optional[dict]:
    """
    解析 Mosaic 认知跟踪输出（选项格式 A/B/C/D/E/F）

    Returns:
        {
            'target_status': str,
            'environment_status': List[str],
            'bbox': List[float],  # XYWH format
            'cognition_chain': str,
            'confidence': float
        } 或 None
    """
    # 选项映射
    status_map = {
        'A': 'normal',
        'B': 'partially_occluded',
        'C': 'fully_occluded',
        'D': 'out_of_view',
        'E': 'disappeared',
        'F': 'reappeared'
    }

    env_map = {
        'A': 'normal',
        'B': 'low_light',
        'C': 'high_light',
        'D': 'motion_blur',
        'E': 'scene_change',
        'F': 'viewpoint_change',
        'G': 'scale_change',
        'H': 'crowded',
        'I': 'background_clutter'
    }

    try:
        t = strip_code_fence(text)
        data = json.loads(t)

        # 解析 target_status（选项格式）
        target_option = data.get('target_status', 'A')
        target_status = status_map.get(target_option, 'normal')

        # 解析 environment_status（选项列表）
        env_options = data.get('environment_status', ['A'])
        if not isinstance(env_options, list):
            env_options = ['A']
        env_status = [env_map.get(opt, 'normal') for opt in env_options if opt in env_map]
        if not env_status:
            env_status = ['normal']

        # 解析 bbox
        bbox_xyxy = None
        if 'bbox' in data:
            b = safe_float_list(data['bbox'])
            if b and b != [0, 0, 0, 0]:
                b = convert_to_pixel_bbox(b, img_width, img_height)
                bbox_xyxy = fix_and_clip_bbox(b, img_width, img_height)

        # 验证 bbox 与 target_status 的一致性
        visible_status = ['normal', 'partially_occluded', 'fully_occluded', 'reappeared']
        if target_status in visible_status and bbox_xyxy is None:
            return None

        if target_status not in visible_status and bbox_xyxy is not None:
            bbox_xyxy = None

        bbox_xywh = xyxy_to_xywh(bbox_xyxy) if bbox_xyxy else [0, 0, 0, 0]

        cognition_chain = data.get('cognition_chain', '')
        if not isinstance(cognition_chain, str) or not cognition_chain.strip():
            cognition_chain = data.get('tracking_evidence', '')

        if (not isinstance(cognition_chain, str) or not cognition_chain.strip()) and \
                isinstance(data.get('memory_update'), dict):
            cognition_chain = (
                data['memory_update'].get('cognition_chain', '')
                or data['memory_update'].get('story', '')
            )

        result = {
            'target_status': target_status,
            'environment_status': env_status,
            'bbox': bbox_xywh,
            'cognition_chain': cognition_chain if isinstance(cognition_chain, str) else '',
            'confidence': float(data.get('confidence', 0.5))
        }

        return result

    except Exception:
        pass

    return None

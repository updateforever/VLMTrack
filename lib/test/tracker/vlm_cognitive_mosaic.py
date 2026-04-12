"""
Qwen VLM Tracker - 认知跟踪（Mosaic 版本）

通过历史帧拼接提供更丰富的时序信息：
    1. 初始化: 从初始帧生成语义记忆
    2. 跟踪: 拼接图(初始帧+历史帧+上一帧) + 当前帧 → 结构化状态判断 + bbox + 认知推理

核心改进（Mosaic 版本）：
    - 历史帧拼接：初始帧(绿框) + 历史高质量帧(蓝框) + 上一帧(红框，任意状态)
    - 上一帧始终更新：提供完整时序信息，包括失败案例
    - 历史帧只保留可见：提供可靠的外观参考
"""
from lib.test.tracker.basetracker import BaseTracker
from lib.test.evaluation.environment import env_settings
import cv2
import numpy as np
import os
from typing import List, Dict, Optional, Tuple

from lib.test.tracker.vlm_engine import VLMEngine
from lib.test.tracker.vlm_utils import (
    parse_mosaic_output, xyxy_to_xywh, draw_bbox,
    parse_memory_state, parse_init_cognition
)
from lib.test.tracker.prompts import get_prompt

# 默认记忆结构（简化为单一 cognition_chain）
_EMPTY_MEMORY = {"cognition_chain": "", "last_update": 0}


class VLMCognitiveMosaic(BaseTracker):
    """
    认知VLM跟踪器（Mosaic 版本）

    跟踪范式:
        初始化: 用VLM分析初始帧，生成语义记忆（叙事）
        跟踪: 拼接图(初始+历史+上一帧) + 当前帧 → 结构化状态判断 + bbox + 认知推理

    拼接策略:
        - 初始帧(绿框): 固定，提供最干净的外观参考
        - 历史帧(红框): 滑动窗口，只保留可见时的高质量帧
        - 上一帧(红框): 始终更新，提供完整时序（包括失败案例）
    """

    def __init__(self, params, dataset_name):
        super().__init__(params)
        self.params = params
        self.dataset_name = dataset_name

        # VLM引擎
        self.vlm = VLMEngine(params)

        # 跟踪状态
        self.state: Optional[List[float]] = None
        self.language_description: str = "the target object"

        # 语义记忆库（认知推理链）
        self.memory: Dict = dict(_EMPTY_MEMORY)

        # 认知跟踪状态
        self.target_status: str = "A"  # 选项格式
        self.environment_status: List[str] = ["A"]
        self.cognition_chain: str = ""
        self.tracking_history: List[Dict] = []

        # Mosaic 专用：历史帧管理
        self.init_frame: Optional[Tuple] = None  # (image, bbox, status)
        self.history_buffer: List[Tuple] = []    # [(frame_id, image, bbox, status)]
        self.prev_frame: Optional[Tuple] = None  # (image, bbox, status)

        # 配置
        self.buffer_size = getattr(params, 'history_buffer_size', 3)
        self.sample_interval = getattr(params, 'sample_interval', 30)
        self.track_prompt = getattr(params, 'track_prompt', 'cognitive_mosaic')
        self.init_prompt = getattr(params, 'init_prompt', 'init_cognition_mosaic')
        self.debug = getattr(params, 'debug', 0)

        self.frame_id = 0
        self.seq_name = None
        self.vis_dir = None

    def initialize(self, image, info: dict):
        """初始化 - 生成初始认知结果并保存初始帧"""
        self.frame_id = 0
        self.state = list(info['init_bbox'])

        self.seq_name = info.get('seq_name', 'unknown')
        self.run_tag = info.get('run_tag', None)
        lang = info.get('init_nlp', None)
        self.language_description = lang if lang else "the target object"

        # 保存初始帧（使用完整状态名）
        self.init_frame = (image.copy(), list(info['init_bbox']), "normal")
        self.prev_frame = (image.copy(), list(info['init_bbox']), "normal")

        # 生成初始认知
        self._generate_initial_memory(image, self.state)

        if self.debug >= 2:
            env = env_settings()
            vis_root = os.path.join(env.results_path, 'vlm_cognitive_mosaic', 'vis')
            if self.run_tag:
                self.vis_dir = os.path.join(vis_root, self.run_tag, self.seq_name)
            else:
                self.vis_dir = os.path.join(vis_root, self.seq_name)
            os.makedirs(self.vis_dir, exist_ok=True)

        if self.debug >= 1:
            print(f"[CognitiveMosaic] Init '{self.seq_name}' | "
                  f"buffer_size={self.buffer_size}, sample_interval={self.sample_interval}")

    def _generate_initial_memory(self, image: np.ndarray, bbox: List[float]):
        """调用VLM生成初始认知结果"""
        img_with_box = draw_bbox(image, bbox, color=(0, 255, 0))
        prompt = get_prompt(self.init_prompt, target_description=self.language_description)

        try:
            output = self.vlm.infer([img_with_box], prompt)

            # 1) Mosaic 专用初始化：优先解析 init_cognition
            cognition = parse_init_cognition(output)

            # 2) 兼容旧格式：appearance/motion/context -> cognition_chain
            if not cognition:
                parsed = parse_memory_state(output)
                if parsed and 'appearance' in parsed:
                    desc = (self.language_description or "").strip()
                    if desc and desc.lower() not in ("the target object", "target object"):
                        parsed['appearance'] = f"{desc}; {parsed.get('appearance', '')}".strip("; ")
                    cognition = f"{parsed['appearance']}. {parsed.get('motion', '')}. {parsed.get('context', '')}".strip()

            if cognition:
                self.memory = {"cognition_chain": cognition, "last_update": 0}
            else:
                raise ValueError("cannot parse init cognition from output")
        except Exception as e:
            if self.debug >= 1:
                print(f"[CognitiveMosaic] Memory init failed ({e}), using description fallback")
            self.memory = {
                "cognition_chain": f"The target is {self.language_description}.",
                "last_update": 0
            }

    def _create_mosaic(self, frames: List[Tuple], target_height: int = 240) -> np.ndarray:
        """
        创建历史帧拼接图

        Args:
            frames: [(image, bbox, status, frame_id, is_gt), ...]
            target_height: 目标高度（默认240，节省token）

        Returns:
            拼接后的图像（横向排列，带标注）
        """
        # 状态映射：选项 → 完整单词 → 显示文本
        option_to_status = {
            'A': 'normal', 'B': 'partially_occluded', 'C': 'fully_occluded',
            'D': 'out_of_view', 'E': 'disappeared', 'F': 'reappeared'
        }
        status_display = {
            'normal': 'Normal',
            'partially_occluded': 'Part-Occl',
            'fully_occluded': 'Full-Occl',
            'out_of_view': 'Out-View',
            'disappeared': 'Disappeared',
            'reappeared': 'Reappeared'
        }

        panels = []
        gap_width = 5  # 图片间隔

        for idx, (img, bbox, status, frame_id, is_gt) in enumerate(frames):
            # 修复颜色通道（OpenCV 是 BGR，需要转为 RGB）
            # if len(img.shape) == 3 and img.shape[2] == 3:
            #     panel = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
            # else:
            panel = img.copy()

            # 绘制边界框（注意：输入bbox是xywh格式，需转为xyxy）
            color = (0, 255, 0) if is_gt else (255, 0, 0)  # 绿色=GT，红色=预测
            if bbox != [0, 0, 0, 0]:
                x, y, w, h = [int(v) for v in bbox]
                x1, y1, x2, y2 = x, y, x + w, y + h
                cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)

            # 调整大小
            h, w = panel.shape[:2]
            scale = target_height / h
            new_w = int(w * scale)
            panel = cv2.resize(panel, (new_w, target_height))

            # 添加顶部标注栏（30像素高，单行显示）
            header_h = 30
            header = np.ones((header_h, new_w, 3), dtype=np.uint8) * 255

            # 解析状态（支持选项格式和完整单词）
            status_word = option_to_status.get(status, status)
            status_text = status_display.get(status_word, status_word)
            box_type = "GT" if is_gt else "Pred"

            # 单行显示：#帧号 (类型) - 状态
            text = f"#{frame_id} ({box_type}) - {status_text}"
            cv2.putText(header, text, (5, 20),
                       cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 0), 1)

            # 拼接标注栏和图像
            panel_with_header = np.vstack([header, panel])
            panels.append(panel_with_header)

            # 添加间隔（除了最后一帧）
            if idx < len(frames) - 1:
                gap = np.ones((target_height + header_h, gap_width, 3), dtype=np.uint8) * 200
                panels.append(gap)

        # 横向拼接
        mosaic = np.hstack(panels)

        # 保持 RGB 格式供 VLM 使用（不需要转回 BGR）
        return mosaic

    def track(self, image, info: dict = None):
        """认知跟踪（Mosaic 版本）"""
        # 关键帧模式下 track() 只在关键帧调用，不能用“调用次数”当帧号。
        # 优先使用外部传入的真实序列帧号（running.py 中的 frame_num）。
        if info is not None and 'frame_num' in info:
            self.frame_id = int(info['frame_num'])
        else:
            self.frame_id += 1
        H, W = image.shape[:2]

        try:
            # 1. 构建拼接图
            frames_to_concat = []

            # 初始帧（绿框，GT）
            frames_to_concat.append((
                self.init_frame[0], self.init_frame[1], self.init_frame[2],
                0, True  # frame_id=0, is_gt=True
            ))

            # 历史帧（红框，预测值）
            for fid, img, bbox, status in self.history_buffer:
                frames_to_concat.append((
                    img, bbox, status,
                    fid, False  # is_gt=False
                ))

            mosaic = self._create_mosaic(frames_to_concat)

            # DEBUG: 可视化 mosaic（在调试控制台运行）
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(20, 5))
            # plt.imshow(mosaic)  # mosaic 已经是 RGB 格式，直接显示
            # plt.title(f"Frame {self.frame_id} Mosaic")
            # plt.axis('off')
            # plt.tight_layout()
            # plt.show()
            # print(f"Mosaic shape: {mosaic.shape}")
            # for i, (img, bbox, status, fid, is_gt) in enumerate(frames_to_concat):
            #     print(f"  Frame #{i}: fid={fid}, bbox={bbox}, status={status}, is_gt={is_gt}")

            # 2. VLM 推理
            prompt_kwargs = dict(
                memory_cognition=self.memory['cognition_chain'],
                language_description=self.language_description,
                num_history_frames=len(self.history_buffer),
            )

            # 消融选项：在 prompt 中显式写入初始 GT 框坐标（归一化到 [0, 999]）
            if getattr(self.params, 'use_init_bbox_ref', False) and self.init_frame is not None:
                ix, iy, iw, ih = self.init_frame[1]
                h0, w0 = self.init_frame[0].shape[:2]
                init_bbox_1000 = [
                    int(max(0, min(999, ix / w0 * 1000))),
                    int(max(0, min(999, iy / h0 * 1000))),
                    int(max(0, min(999, (ix + iw) / w0 * 1000))),
                    int(max(0, min(999, (iy + ih) / h0 * 1000))),
                ]
                prompt_kwargs['init_bbox_1000'] = init_bbox_1000

            prompt = get_prompt(self.track_prompt, **prompt_kwargs)

            output = self.vlm.infer([mosaic, image], prompt)

            if self.debug >= 1:
                kf_tag = "[KF]" if (info and info.get('is_keyframe')) else ""
                print(f"[CognitiveMosaic] Frame {self.frame_id}{kf_tag} | "
                      f"History buffer: {len(self.history_buffer)} frames")
                print(f"  Raw output: {output[:150]}...")

            # 3. 解析结构化输出（新格式：选项 A/B/C）
            result = parse_mosaic_output(output, W, H)

            if result:
                if self.debug >= 1:
                    print(f"  Status: {result['target_status']}")
                    print(f"  Environment: {result['environment_status']}")
                    print(f"  Confidence: {result['confidence']:.2f}")
                    print(f"  Cognition: {result['cognition_chain'][:80]}...")

                # 更新状态
                self.state = result['bbox']
                self.target_status = result['target_status']
                self.environment_status = result['environment_status']
                self.cognition_chain = result['cognition_chain']

                # 4. 更新历史帧策略
                # 上一帧：始终更新（包括失败案例）
                self.prev_frame = (image.copy(), result['bbox'], result['target_status'])

                # 历史帧：只保存目标可见的帧（有有效 bbox 的），确保 mosaic 有视觉参考价值
                # 目标消失帧（out_of_view/disappeared）没有 bbox，放进历史缓冲只会浪费位置
                target_visible = result['target_status'] in ('normal', 'partially_occluded', 'reappeared') and result['bbox'] != [0, 0, 0, 0]

                if target_visible:
                    should_save = False
                    is_kf = info and info.get('is_keyframe', False)

                    if is_kf:
                        should_save = True
                    elif not self.history_buffer:
                        should_save = True
                    else:
                        last_frame_id = self.history_buffer[-1][0]
                        if self.frame_id - last_frame_id >= self.sample_interval:
                            should_save = True

                    if should_save:
                        self.history_buffer.append((
                            self.frame_id, image.copy(), result['bbox'], result['target_status']
                        ))
                        # FIFO 队列
                        if len(self.history_buffer) > self.buffer_size:
                            self.history_buffer.pop(0)

                # 保存跟踪历史
                self.tracking_history.append({
                    'frame_id': self.frame_id,
                    'target_status': result['target_status'],
                    'environment_status': result['environment_status'],
                    'bbox': result['bbox'],
                    'confidence': result['confidence'],
                    'cognition_chain': result['cognition_chain']
                })

                # 当前认知推理链既是证据，也是后续帧的记忆输入
                if result['cognition_chain']:
                    self.memory = {
                        "cognition_chain": result['cognition_chain'],
                        "last_update": self.frame_id
                    }

                if self.debug >= 2:
                    self._save_vis(mosaic, image, result)

                # 保存当前帧的完整结果（用于返回）
                current_result = result

            else:
                # Parse失败
                if self.debug >= 1:
                    print(f"[CognitiveMosaic] Frame {self.frame_id}: Parse failed, marking as absent")
                self.state = [0, 0, 0, 0]
                self.target_status = "E"  # disappeared
                self.environment_status = ["A"]
                self.cognition_chain = ""

                # 上一帧仍然更新
                self.prev_frame = (image.copy(), [0, 0, 0, 0], "E")

                # 保存失败结果
                current_result = {
                    'bbox': [0, 0, 0, 0],
                    'target_status': 'E',
                    'environment_status': ['A'],
                    'cognition_chain': '',
                    'confidence': 0.0
                }

        except Exception as e:
            if self.debug >= 1:
                print(f"[CognitiveMosaic] Frame {self.frame_id} error: {e}")
            import traceback
            traceback.print_exc()
            self.state = [0, 0, 0, 0]
            self.target_status = "E"
            self.environment_status = ["A"]
            self.cognition_chain = ""

            # 保存错误结果
            current_result = {
                'bbox': [0, 0, 0, 0],
                'target_status': 'E',
                'environment_status': ['A'],
                'cognition_chain': '',
                'confidence': 0.0
            }

        # 返回完整跟踪数据（用于保存）
        out = {"target_bbox": self.state}

        # 附加 VLM 元数据
        out['vlm_metadata'] = {
            'target_status': current_result.get('target_status', 'E'),
            'environment_status': current_result.get('environment_status', ['A']),
            'cognition_chain': current_result.get('cognition_chain', ''),
            'confidence': float(current_result.get('confidence', 0.0))
        }

        return out

    def _save_vis(self, mosaic: np.ndarray, current: np.ndarray, result: dict):
        """保存可视化（增强版：叠加认知链文本）"""
        if not self.vis_dir:
            return

        bbox = result['bbox']
        if bbox != [0, 0, 0, 0]:
            result_img = draw_bbox(current, bbox, color=(255, 0, 0))
        else:
            result_img = current.copy()

        # mosaic/current/result_img 都按 RGB 处理，写盘前再统一转 BGR
        # 调整 mosaic 高度与 current 一致
        h_cur = current.shape[0]
        h_mos, w_mos = mosaic.shape[:2]
        scale = h_cur / h_mos
        mosaic_resized = cv2.resize(mosaic, (int(w_mos * scale), h_cur))

        # 拼接上半部分：mosaic | result_img
        combined_top = np.hstack([mosaic_resized, result_img])

        # ========== 增强：添加文本信息面板 ==========
        # 创建文本面板（宽度匹配 combined_top）
        panel_height = 180
        panel_width = combined_top.shape[1]  # 修复：使用 combined_top 的宽度
        text_panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 240

        # 状态映射
        status_map = {
            'A': 'normal', 'B': 'partially_occluded', 'C': 'fully_occluded',
            'D': 'out_of_view', 'E': 'disappeared', 'F': 'reappeared'
        }
        env_map = {
            'A': 'normal', 'B': 'low_light', 'C': 'high_light', 'D': 'motion_blur',
            'E': 'scene_change', 'F': 'viewpoint_change', 'G': 'scale_change',
            'H': 'crowded', 'I': 'background_clutter'
        }

        # 解析状态
        target_status = result.get('target_status', 'A')
        target_status_text = status_map.get(target_status, target_status)
        env_status = result.get('environment_status', ['A'])
        env_status_text = [env_map.get(e, e) for e in env_status]
        confidence = result.get('confidence', 0.0)

        # 第1行：帧号和状态
        line1 = f"Frame {self.frame_id}  |  Status: {target_status}-{target_status_text}  |  Conf: {confidence:.2f}"
        cv2.putText(text_panel, line1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # 第2行：环境状态
        line2 = f"Environment: {', '.join(env_status_text)}"
        cv2.putText(text_panel, line2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)

        # 第3-6行：认知链（自动换行）
        cognition = result.get('cognition_chain', '')
        cv2.putText(text_panel, "Cognition Chain:", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 简单的文本换行（按字符数）
        max_chars_per_line = int(panel_width / 8)  # 根据面板宽度动态计算
        words = cognition.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars_per_line:
                current_line += word + " "
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        if current_line:
            lines.append(current_line.strip())

        # 最多显示3行
        for i, line in enumerate(lines[:3]):
            cv2.putText(text_panel, line, (10, 100 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)

        # 如果文本太长，显示省略号
        if len(lines) > 3:
            cv2.putText(text_panel, "...", (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

        # 拼接：上半部分 + 文本面板（垂直）
        combined = np.vstack([combined_top, text_panel])
        # ==========================================
        _, w = combined.shape[:2]
        pad = np.ones((120, w, 3), dtype=np.uint8) * 255
        font, sc = cv2.FONT_HERSHEY_SIMPLEX, 0.4

        cv2.putText(pad, f"Frame: {self.frame_id} | Status: {result['target_status']}",
                   (8, 24), font, sc, (0, 0, 0), 1)
        cv2.putText(pad, f"Environment: {', '.join(result['environment_status'])}",
                   (8, 50), font, sc, (0, 0, 0), 1)
        cv2.putText(pad, f"Confidence: {result['confidence']:.2f}",
                   (8, 76), font, sc, (0, 0, 0), 1)
        cognition = result['cognition_chain'][:100]
        cv2.putText(pad, f"Cognition: {cognition}",
                   (8, 102), font, sc, (255, 0, 0), 1)

        final = np.vstack([combined, pad])
        save_path = os.path.join(self.vis_dir, f"{self.frame_id:06d}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))


def get_tracker_class():
    return VLMCognitiveMosaic

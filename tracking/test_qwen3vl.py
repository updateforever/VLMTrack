"""
Qwen3VL Tracker 快速测试脚本

用法:
    # 快速测试 (dummy图像)
    python tracking/test_qwen3vl.py
    
    # 在视频上测试
    python tracking/test_qwen3vl.py --video your_video.mp4
    
    # 使用标准测试脚本 (推荐)
    python tracking/test.py qwen3vl qwen25vl_3b --dataset tnl2k --threads 0
    
注意:
    - VLM不支持多线程,必须使用 --threads 0
    - 首次运行会下载模型 (~6GB)
    - 推理速度约 1-2 FPS
"""
import os
import sys
import argparse
import cv2
import numpy as np

# Add project root to path
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)


def test_with_dummy_images():
    """使用dummy图像快速测试模型加载和推理"""
    print("=" * 60)
    print("Qwen3VL Tracker - Quick Test")
    print("=" * 60)
    
    from lib.test.parameter.qwen3vl import parameters
    from lib.test.tracker.qwen3vl import QWEN3VL
    
    # 加载参数
    params = parameters('qwen25vl_3b')
    params.debug = 1  # 打印调试信息
    
    print(f"\n[1] Loading model: {params.model_name}")
    print("    This may take a while for the first time...")
    
    # 创建tracker
    tracker = QWEN3VL(params, dataset_name='test')
    print("[2] Model loaded successfully!")
    
    # 创建测试图像 - 红色矩形目标
    print("\n[3] Creating test images...")
    
    # Template: 红色矩形在位置 (100, 100)
    template = np.ones((480, 640, 3), dtype=np.uint8) * 200  # 灰色背景
    cv2.rectangle(template, (100, 100), (200, 200), (0, 0, 255), -1)  # 红色目标
    
    # Search: 红色矩形移动到 (150, 120)
    search = np.ones((480, 640, 3), dtype=np.uint8) * 200
    cv2.rectangle(search, (150, 120), (250, 220), (0, 0, 255), -1)  # 移动后
    
    # 转换为RGB (tracker期望RGB输入)
    template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    search_rgb = cv2.cvtColor(search, cv2.COLOR_BGR2RGB)
    
    # 初始化
    init_info = {
        'init_bbox': [100, 100, 100, 100],  # [x, y, w, h]
        'init_nlp': 'a red rectangle'
    }
    
    print(f"[4] Initializing tracker...")
    print(f"    Template bbox: {init_info['init_bbox']}")
    print(f"    Description: {init_info['init_nlp']}")
    
    tracker.initialize(template_rgb, init_info)
    
    # 跟踪
    print(f"\n[5] Running tracking...")
    result = tracker.track(search_rgb)
    
    print(f"\n[6] Results:")
    print(f"    Predicted bbox: {result['target_bbox']}")
    print(f"    Confidence: {result['best_score']:.2f}")
    
    # 期望结果
    expected = [150, 120, 100, 100]
    
    # 计算IoU
    def calc_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0
    
    iou = calc_iou(expected, result['target_bbox'])
    print(f"    Expected bbox: {expected}")
    print(f"    IoU with expected: {iou:.2f}")
    
    if iou > 0.5:
        print("\n✅ Test PASSED!")
    else:
        print("\n⚠️ Test completed but IoU is low. Check model output.")
    
    print("=" * 60)
    return result


def test_on_video(video_path, init_bbox=None):
    """在视频上测试tracker"""
    from lib.test.parameter.qwen3vl import parameters
    from lib.test.tracker.qwen3vl import QWEN3VL
    
    params = parameters('qwen25vl_3b')
    params.debug = 1
    
    print(f"Loading tracker...")
    tracker = QWEN3VL(params, dataset_name='video')
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        return
    
    # 选择目标
    if init_bbox is None:
        print("Select target and press ENTER...")
        bbox = cv2.selectROI("Select Target", frame, fromCenter=False)
        cv2.destroyWindow("Select Target")
        init_bbox = list(bbox)
    
    # 初始化
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    init_info = {
        'init_bbox': init_bbox,
        'init_nlp': 'the selected object'
    }
    
    tracker.initialize(frame_rgb, init_info)
    print(f"Initialized with bbox: {init_bbox}")
    
    # 跟踪循环
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        result = tracker.track(frame_rgb)
        bbox = [int(v) for v in result['target_bbox']]
        
        # 可视化
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Frame {frame_id} Score: {result['best_score']:.2f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Qwen3VL Tracking', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Qwen3VL Tracker')
    parser.add_argument('--video', type=str, default=None, help='Video file path')
    parser.add_argument('--model', type=str, default='qwen25vl_3b', help='Model config')
    parser.add_argument('--debug', type=int, default=1, help='Debug level (0/1/2)')
    
    args = parser.parse_args()
    
    if args.video:
        test_on_video(args.video)
    else:
        test_with_dummy_images()

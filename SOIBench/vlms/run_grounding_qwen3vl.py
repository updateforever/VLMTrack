# -*- coding: utf-8 -*-
"""
SOIBench/vlms/run_grounding_qwen3vl.py
Grounding 主流程脚本
功能：
1）读取描述 jsonl，修复图像路径
2）调用 qwen3vl_infer.py 中的本地或 API 引擎推理
3）解析输出 bbox，并统一为像素坐标
4）保存 pred.jsonl，支持断点续跑
5）可选保存可视化结果
"""

import argparse
import glob
import json
import os
import re
import traceback
from pathlib import Path

from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageColor

from qwen3vl_infer import Qwen3VLLocalEngine, qwen3vl_api_chat


DATASET_IMAGE_ROOTS = {
    "lasot": "/home/member/data1/DATASETS_PUBLIC/LaSOT/LaSOTBenchmark",
    "mgit":  "/home/member/data1/DATASETS_PUBLIC/MGIT/VideoCube/MGIT-Test/data/test",
    "tnl2k": "/home/member/data1/DATASETS_PUBLIC/TNL2K/TNL2K_CVPR2021/test"
}


_ADDITIONAL_COLORS = [name for (name, _) in ImageColor.colormap.items()]


def plot_bounding_boxes(im: Image.Image, bboxes, save_path: str):
    """
    在图上画 bbox
    bboxes: List[List[float]]，每个为 [x1,y1,x2,y2] 像素坐标
    """
    if not bboxes:
        return

    img = im.copy()
    draw = ImageDraw.Draw(img)
    colors = ["red", "green", "blue", "yellow", "orange", "pink", "purple"] + _ADDITIONAL_COLORS

    try:
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
    except Exception:
        font = ImageFont.load_default()

    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=3)
        draw.text((x1 + 4, y1 + 4), f"Pred-{i}", fill=color, font=font)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)


def _strip_code_fence(text: str) -> str:
    """
    去掉 ```json / ``` 等包裹
    """
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _safe_float_list(x):
    """
    尝试把输入转换成长度为 4 的 float list
    """
    if isinstance(x, (list, tuple)) and len(x) == 4:
        try:
            return [float(v) for v in x]
        except Exception:
            return None
    return None


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _fix_and_clip_bbox(b, w, h):
    """
    修正顺序并裁剪到图像范围
    """
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


def _convert_to_pixel_bbox(b, w, h):
    """
    支持三类坐标体系并统一成像素坐标
    1）0 到 1 归一化
    2）0 到 1000 归一化
    3）像素坐标
    """
    x1, y1, x2, y2 = b
    maxv = max(x1, y1, x2, y2)
    minv = min(x1, y1, x2, y2)

    if 0.0 <= minv and maxv <= 1.0:
        return [x1 * w, y1 * h, x2 * w, y2 * h]

    if 0.0 <= minv and maxv <= 1000.0:
        return [(x1 / 1000.0) * w, (y1 / 1000.0) * h, (x2 / 1000.0) * w, (y2 / 1000.0) * h]

    return [x1, y1, x2, y2]


def extract_bboxes_from_model_output(text: str, img_width: int, img_height: int):
    """
    bbox 解析
    支持：
    1）JSON: {"bbox_2d":[...]} 或 [{"bbox_2d":[...]}]
    2）JSON: [x1,y1,x2,y2]
    3）文本中包含多个 [x1,y1,x2,y2]
    4）兼容 0 到 1，0 到 1000，像素坐标
    返回：List[[x1,y1,x2,y2]]，像素坐标
    """
    raw = text or ""
    t = _strip_code_fence(raw)
    if not t:
        return []

    bboxes = []

    try:
        data = json.loads(t)

        if isinstance(data, dict):
            if "bbox_2d" in data:
                b = _safe_float_list(data["bbox_2d"])
                if b:
                    b = _convert_to_pixel_bbox(b, img_width, img_height)
                    bboxes.append(_fix_and_clip_bbox(b, img_width, img_height))

            if "bboxes" in data and isinstance(data["bboxes"], list):
                for it in data["bboxes"]:
                    if isinstance(it, dict) and "bbox_2d" in it:
                        b = _safe_float_list(it["bbox_2d"])
                    else:
                        b = _safe_float_list(it)
                    if b:
                        b = _convert_to_pixel_bbox(b, img_width, img_height)
                        bboxes.append(_fix_and_clip_bbox(b, img_width, img_height))

        elif isinstance(data, list):
            if len(data) == 4 and all(isinstance(x, (int, float)) for x in data):
                b = [float(x) for x in data]
                b = _convert_to_pixel_bbox(b, img_width, img_height)
                bboxes.append(_fix_and_clip_bbox(b, img_width, img_height))
            else:
                for it in data:
                    if isinstance(it, dict) and "bbox_2d" in it:
                        b = _safe_float_list(it["bbox_2d"])
                    else:
                        b = _safe_float_list(it)
                    if b:
                        b = _convert_to_pixel_bbox(b, img_width, img_height)
                        bboxes.append(_fix_and_clip_bbox(b, img_width, img_height))

        if bboxes:
            return bboxes

    except Exception:
        pass

    matches = re.findall(r"\[([^\[\]]+)\]", t)
    for m in matches:
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m)
        if len(nums) == 4:
            try:
                b = [float(x) for x in nums]
                b = _convert_to_pixel_bbox(b, img_width, img_height)
                bboxes.append(_fix_and_clip_bbox(b, img_width, img_height))
            except Exception:
                continue

    return bboxes


def load_and_fix_paths(jsonl_path: str, dataset_name: str):
    """
    读取描述 jsonl，并把 image_path 修复为绝对路径
    抽取 output-en 的 level1 到 level4 拼成描述文本
    """
    image_root = DATASET_IMAGE_ROOTS.get(dataset_name)
    if not image_root:
        return []

    valid = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)

            if item.get("status") == "skip":
                continue

            output_en = item.get("output-en", {}) or {}
            desc_parts = []
            for k in ["level1", "level2", "level3", "level4"]:
                v = (output_en.get(k, "") or "").strip()
                if v:
                    desc_parts.append(v)

            full_desc = " ".join(desc_parts).strip()
            if not full_desc:
                continue

            rel = item.get("image_path", "")
            if not rel:
                continue
            if rel.startswith("/"):
                rel = rel[1:]
            abs_path = os.path.join(image_root, rel)

            if os.path.exists(abs_path):
                valid.append({
                    "original_item": item,
                    "image_path": abs_path,
                    "text_prompt": full_desc,
                    "dataset_name": dataset_name,
                    "frame_idx": item.get("frame_idx", "unknown"),
                })
            elif os.path.exists(os.path.join(image_root, rel[6:10], rel)):
                abs_path = os.path.join(image_root, rel[6:10], rel)
                valid.append({
                    "original_item": item,
                    "image_path": abs_path,
                    "text_prompt": full_desc,
                    "dataset_name": dataset_name,
                    "frame_idx": item.get("frame_idx", "unknown"),
                })
            elif os.path.exists(os.path.join(image_root, rel.split('/')[1], 'imgs', rel.split('/')[2][1:])):
                abs_path = os.path.join(image_root, rel.split('/')[1], 'imgs', rel.split('/')[2][1:])
                valid.append({
                    "original_item": item,
                    "image_path": abs_path,
                    "text_prompt": full_desc,
                    "dataset_name": dataset_name,
                    "frame_idx": item.get("frame_idx", "unknown"),
                })
            elif os.path.exists(os.path.join(image_root, rel.split('/')[1], 'imgs', rel.split('/')[2])):
                abs_path = os.path.join(image_root, rel.split('/')[1], 'imgs', rel.split('/')[2])
                valid.append({
                    "original_item": item,
                    "image_path": abs_path,
                    "text_prompt": full_desc,
                    "dataset_name": dataset_name,
                    "frame_idx": item.get("frame_idx", "unknown"),
                })
            else:
                print('error for img load')

    return valid


def build_prompt(description: str) -> str:
    """
    Grounding prompt，强约束输出格式
    坐标强制使用 0 到 1000 归一化，减少模型输出形态漂移
    """
    return (
        "You are a visual grounding model. Given an image and a target description, output the target bounding box.\n"
        f"Target description: {description}\n"
        "Locate the description target, output its bbox coordinates using JSON format."
    )


def _count_lines(path: str) -> int:
    """
    统计文件行数，用于断点续跑
    """
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="local", choices=["local", "api"])
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=256)

    parser.add_argument("--api_model_name", type=str, default="qwen-vl-max")
    parser.add_argument("--api_base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api_key_env", type=str, default="sk-61547e720ce8407aa44f4511051903b0")
    parser.add_argument("--api_temperature", type=float, default=0.1)
    parser.add_argument("--api_max_tokens", type=int, default=256)
    parser.add_argument("--api_retries", type=int, default=3)

    parser.add_argument("--exp_tag", type=str, default="run")
    parser.add_argument("--save_debug_vis", action="store_true")

    parser.add_argument("--lasot_dir", type=str, default="/home/member/data2/wyp/SUTrack/SOIBench/data/test/lasot")
    parser.add_argument("--mgit_dir", type=str, default="/home/member/data2/wyp/SUTrack/SOIBench/data/test/mgit")
    parser.add_argument("--tnl2k_dir", type=str, default="/home/member/data2/wyp/SUTrack/SOIBench/data/test/tnl2k")

    parser.add_argument("--output_root", type=str, default="./results")

    args = parser.parse_args()

    engine = None
    if args.mode == "local":
        if not args.model_path:
            raise ValueError("mode=local 时必须提供 --model_path")
        engine = Qwen3VLLocalEngine(args.model_path)

    tasks = []
    if args.lasot_dir:
        tasks.append(("lasot", args.lasot_dir))
    if args.mgit_dir:
        tasks.append(("mgit", args.mgit_dir))
    if args.tnl2k_dir:
        tasks.append(("tnl2k", args.tnl2k_dir))

    if not tasks:
        print("未指定任何数据目录")
        return

    for dataset_name, jsonl_dir in tasks:
        out_dir = os.path.join(args.output_root, dataset_name, f"{args.mode}_{args.exp_tag}")
        os.makedirs(out_dir, exist_ok=True)

        vis_dir = None
        if args.save_debug_vis:
            vis_dir = os.path.join(out_dir, "vis_debug")
            os.makedirs(vis_dir, exist_ok=True)

        jsonl_files = sorted(glob.glob(os.path.join(jsonl_dir, "*.jsonl")))
        if not jsonl_files:
            print(f"目录为空: {jsonl_dir}")
            continue

        for jsonl_file in tqdm(jsonl_files, desc=f"处理 {dataset_name}", dynamic_ncols=True):
            seq_name = Path(jsonl_file).stem.replace("_descriptions", "").replace("_done", "")
            save_path = os.path.join(out_dir, f"{seq_name}_pred.jsonl")
            err_path = os.path.join(out_dir, f"{seq_name}_errors.jsonl")

            processed = _count_lines(save_path)
            samples = load_and_fix_paths(jsonl_file, dataset_name)

            if processed >= len(samples):
                continue

            pending = samples[processed:]

            if not os.path.exists(save_path):
                with open(save_path, "w", encoding="utf-8"):
                    pass

            for s in tqdm(pending, desc=f"序列 {seq_name}", leave=False, dynamic_ncols=True):
                img_path = s["image_path"]
                # try:
                
                prompt = build_prompt(s["text_prompt"])

                if args.mode == "local":
                    raw_out = engine.chat(img_path, prompt, max_new_tokens=args.max_new_tokens)
                else:
                    raw_out = qwen3vl_api_chat(
                        image_path=img_path,
                        prompt=prompt,
                        model_name=args.api_model_name,
                        base_url=args.api_base_url,
                        api_key=args.api_key_env,
                        temperature=args.api_temperature,
                        max_tokens=args.api_max_tokens,
                        retries=args.api_retries,
                        retry_sleep=1.0,
                    )

                if not raw_out:
                    record = s["original_item"].copy()
                    record["model_raw_response"] = raw_out
                    record["parsed_bboxes"] = []
                    record["parse_status"] = "empty_output"
                    with open(save_path, "a", encoding="utf-8") as f_out:
                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue

                with Image.open(img_path) as img:
                    w, h = img.size
                    parsed = extract_bboxes_from_model_output(raw_out, w, h)

                record = s["original_item"].copy()
                record["model_raw_response"] = raw_out
                record["parsed_bboxes"] = parsed
                record["parse_status"] = "ok" if parsed else "no_bbox_found"

                with open(save_path, "a", encoding="utf-8") as f_out:
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

                if vis_dir and parsed:
                    vis_path = os.path.join(vis_dir, f"{seq_name}_{s['frame_idx']}.jpg")
                    with Image.open(img_path) as img:
                        plot_bounding_boxes(img, parsed, vis_path)

                # except Exception as e:
                #     print(e)
                #     err_rec = {
                #         "image_path": img_path,
                #         "dataset_name": dataset_name,
                #         "seq_name": seq_name,
                #         "frame_idx": s.get("frame_idx", "unknown"),
                #         "error": str(e),
                #         "traceback": traceback.format_exc(),
                #     }
                #     with open(err_path, "a", encoding="utf-8") as f_err:
                #         f_err.write(json.dumps(err_rec, ensure_ascii=False) + "\n")

    print("✅ 全部任务完成")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the CognitiveBench v1 annotation pack.

The output intentionally keeps the classic tracking layout per sequence:
groundtruth.txt, target_status.txt, keyframes.txt, and meta.json.
Images are not copied. Loaders should resolve frames from the original dataset
roots using source_dataset + sequence + split.
"""

import argparse
import json
import math
import os
from pathlib import Path


STATUS_PRESENT = 1
STATUS_ABSENT = 0


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_numeric_rows(path, delimiter=","):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if delimiter:
                parts = [p.strip() for p in line.split(delimiter)]
            else:
                parts = line.split()
            rows.append([float(p) for p in parts if p != ""])
    return rows


def read_numeric_vector(path):
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return vals
    # LaSOT masks are often one comma-separated line; MGIT masks are line-based.
    text = text.replace(",", "\n")
    for token in text.split():
        vals.append(int(float(token)))
    return vals


def write_groundtruth(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            if len(row) < 4:
                raise ValueError(f"Invalid bbox row in {path}: {row}")
            x, y, w, h = row[:4]
            f.write(f"{format_num(x)},{format_num(y)},{format_num(w)},{format_num(h)}\n")


def write_vector(path, vals):
    with open(path, "w", encoding="utf-8") as f:
        for v in vals:
            f.write(f"{int(v)}\n")


def write_keyframes(path, keyframes):
    with open(path, "w", encoding="utf-8") as f:
        for idx in keyframes:
            f.write(f"{int(idx)}\n")


def write_meta(path, meta):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")


def format_num(x):
    if math.isfinite(x) and abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.6f}".rstrip("0").rstrip(".")


def valid_bbox(row):
    if len(row) < 4:
        return False
    x, y, w, h = row[:4]
    return all(math.isfinite(v) for v in (x, y, w, h)) and w > 0 and h > 0


def ensure_aligned(seq, gt, status, keyframes, total_frames):
    n = len(gt)
    if len(status) != n:
        raise ValueError(f"{seq}: target_status length {len(status)} != groundtruth length {n}")
    if total_frames is not None and int(total_frames) != n:
        raise ValueError(f"{seq}: keyframe total_frames {total_frames} != groundtruth length {n}")
    bad = [idx for idx in keyframes if idx < 0 or idx >= n]
    if bad:
        raise ValueError(f"{seq}: keyframes out of range, e.g. {bad[:5]} for length {n}")


def build_lasot(seq, keyframe_data, args):
    class_name = seq.split("-")[0]
    seq_root = Path(args.lasot_root) / class_name / seq
    gt = read_numeric_rows(seq_root / "groundtruth.txt", delimiter=",")
    full_occ = read_numeric_vector(seq_root / "full_occlusion.txt")
    out_view = read_numeric_vector(seq_root / "out_of_view.txt")
    if len(full_occ) != len(gt) or len(out_view) != len(gt):
        raise ValueError(
            f"{seq}: LaSOT status length mismatch "
            f"full_occlusion={len(full_occ)} out_of_view={len(out_view)} gt={len(gt)}"
        )
    status = [
        STATUS_ABSENT if (full_occ[i] != 0 or out_view[i] != 0) else STATUS_PRESENT
        for i in range(len(gt))
    ]
    return gt, status, {
        "target_status_source": "lasot_full_occlusion_or_out_of_view",
        "source_dataset": "lasot",
        "source_split": "test",
    }


def build_mgit(seq, keyframe_data, args):
    seq = normalize_mgit_sequence(seq)
    attr_root = Path(args.mgit_root) / "attribute"
    gt = read_numeric_rows(attr_root / "groundtruth" / f"{seq}.txt", delimiter=",")
    absent = read_numeric_vector(attr_root / "absent" / f"{seq}.txt")
    if len(absent) != len(gt):
        raise ValueError(f"{seq}: MGIT absent length {len(absent)} != gt length {len(gt)}")
    status = [STATUS_ABSENT if v != 0 else STATUS_PRESENT for v in absent]
    return gt, status, {
        "target_status_source": "mgit_absent",
        "source_dataset": "mgit",
        "source_split": "val",
    }


def normalize_mgit_sequence(seq):
    if seq.startswith("frame_"):
        return seq[len("frame_"):]
    return seq


def build_tnl2k(seq, keyframe_data, args):
    seq_root = Path(args.tnl2k_root) / seq
    gt = read_numeric_rows(seq_root / "groundtruth.txt", delimiter=",")
    status = [STATUS_PRESENT if valid_bbox(row) else STATUS_ABSENT for row in gt]
    return gt, status, {
        "target_status_source": "derived_from_gt_bbox",
        "source_dataset": "tnl2k",
        "source_split": "test",
    }


DATASETS = {
    "mgit": {
        "split": "val",
        "builder": build_mgit,
    },
    "lasot": {
        "split": "test",
        "builder": build_lasot,
    },
    "tnl2k": {
        "split": "test",
        "builder": build_tnl2k,
    },
}


def build_dataset(name, args):
    spec = DATASETS[name]
    keyframe_dir = Path(args.keyframe_root) / name / spec["split"]
    if not keyframe_dir.is_dir():
        raise FileNotFoundError(f"Missing keyframe directory: {keyframe_dir}")

    out_split_dir = Path(args.output_root) / "test"
    out_split_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for keyframe_path in sorted(keyframe_dir.glob("*.json")):
        seq = keyframe_path.stem
        out_seq = normalize_mgit_sequence(seq) if name == "mgit" else seq
        keyframe_data = read_json(keyframe_path)
        keyframes = keyframe_data.get("key_frames", [])
        total_frames = keyframe_data.get("total_frames")

        gt, status, meta_extra = spec["builder"](seq, keyframe_data, args)
        ensure_aligned(out_seq, gt, status, keyframes, total_frames)

        seq_out = out_split_dir / out_seq
        if seq_out.exists() and not args.overwrite:
            raise FileExistsError(f"Output sequence exists: {seq_out} (use --overwrite)")
        seq_out.mkdir(parents=True, exist_ok=True)

        write_groundtruth(seq_out / "groundtruth.txt", gt)
        write_vector(seq_out / "target_status.txt", status)
        write_keyframes(seq_out / "keyframes.txt", keyframes)

        meta = {
            "sequence": out_seq,
            "split": "test",
            "num_frames": len(gt),
            "bbox_format": "xywh",
            "frame_index_base": 0,
            "target_status_format": {
                "1": "present",
                "0": "absent",
            },
        }
        meta.update(meta_extra)
        write_meta(seq_out / "meta.json", meta)
        count += 1

    return count


def write_benchmark_meta(args, counts):
    meta = {
        "name": "CognitiveBench",
        "version": "v1",
        "split": "test",
        "bbox_format": "xywh",
        "frame_index_base": 0,
        "target_status_format": {
            "1": "present",
            "0": "absent",
        },
        "source_datasets": ["mgit", "lasot", "tnl2k"],
        "source_splits": {
            "mgit": "val",
            "lasot": "test",
            "tnl2k": "test",
        },
        "sequence_counts": counts,
        "notes": [
            "Images are not copied.",
            "Frame paths are resolved from original dataset roots by source_dataset and sequence.",
            "MGIT uses val because test ground truth is not public.",
        ],
    }
    write_meta(Path(args.output_root) / "benchmark_meta.json", meta)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_root",
        default="/data/DATASETS_PUBLIC/CognitiveBench",
        help="Output annotation pack root.",
    )
    parser.add_argument(
        "--keyframe_root",
        default="/data/DATASETS_PUBLIC/SOIBench/KeyFrame/scene_changes_clip/top_10",
        help="Root containing dataset/split/*.json keyframe files.",
    )
    parser.add_argument("--lasot_root", default="/data/DATASETS_PUBLIC/lasot")
    parser.add_argument("--mgit_root", default="/data/DATASETS_PUBLIC/MGIT")
    parser.add_argument("--tnl2k_root", default="/data/DATASETS_PUBLIC/TNL2K/TNL2K_test_subset")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mgit", "lasot", "tnl2k"],
        choices=sorted(DATASETS.keys()),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    counts = {}
    for name in args.datasets:
        counts[name] = build_dataset(name, args)
        print(f"[CognitiveBench] built {counts[name]} sequences from {name}")

    write_benchmark_meta(args, counts)
    print(f"[CognitiveBench] done: {args.output_root}")


if __name__ == "__main__":
    main()

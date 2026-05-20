import json
import os
from pathlib import Path

import numpy as np

from lib.test.evaluation.data import BaseDataset, Sequence, SequenceList
from lib.test.utils.load_text import load_text


class CognitiveBenchDataset(BaseDataset):
    """
    CognitiveBench v1 evaluation set.

    The benchmark stores annotations only:
        CognitiveBench/test/<sequence>/
            groundtruth.txt
            target_status.txt   # 1=present, 0=absent
            keyframes.txt
            meta.json

    Images are resolved from the original datasets using source_dataset,
    source_split, and sequence in meta.json. The dataset itself is dense;
    sparse VLM execution is controlled by the evaluation keyframe loader.
    """

    def __init__(self, split="test"):
        super().__init__()
        self.split = split
        self.base_path = Path(getattr(
            self.env_settings,
            "cognitivebench_path",
            "/data/DATASETS_PUBLIC/CognitiveBench",
        ))
        self.anno_path = self.base_path / split
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _get_sequence_list(self):
        if not self.anno_path.is_dir():
            raise FileNotFoundError(f"CognitiveBench split directory not found: {self.anno_path}")

        seqs = []
        for seq_dir in sorted(self.anno_path.iterdir()):
            if not seq_dir.is_dir():
                continue
            if (seq_dir / "meta.json").is_file():
                seqs.append(seq_dir.name)
        return seqs

    def _construct_sequence(self, sequence_name):
        seq_dir = self.anno_path / sequence_name
        meta = self._read_meta(seq_dir / "meta.json")
        source_dataset = meta["source_dataset"].lower()
        source_split = meta.get("source_split", self.split)
        source_sequence = meta.get("sequence", sequence_name)

        anno_path = seq_dir / "groundtruth.txt"
        ground_truth_rect = load_text(str(anno_path), delimiter=",", dtype=np.float64)
        ground_truth_rect = ground_truth_rect.reshape(-1, 4)

        status_path = seq_dir / "target_status.txt"
        target_status = load_text(str(status_path), delimiter=",", dtype=np.int64, backend="numpy")
        target_status = np.asarray(target_status).reshape(-1)
        target_visible = target_status == 1
        keyframe_indices = self._read_keyframes(seq_dir / "keyframes.txt")

        if len(target_visible) != len(ground_truth_rect):
            raise ValueError(
                f"CognitiveBench sequence {sequence_name}: target_status length "
                f"{len(target_visible)} != groundtruth length {len(ground_truth_rect)}"
            )

        frames = self._resolve_frames(source_dataset, source_sequence, source_split, len(ground_truth_rect))
        if len(frames) != len(ground_truth_rect):
            raise ValueError(
                f"CognitiveBench sequence {sequence_name}: frame count "
                f"{len(frames)} != groundtruth length {len(ground_truth_rect)}"
            )

        language_query = self._resolve_language(source_dataset, source_sequence)

        seq = Sequence(
            sequence_name,
            frames,
            "cognitivebench",
            ground_truth_rect,
            target_visible=target_visible,
            language_query=language_query,
        )
        seq.source_dataset = source_dataset
        seq.source_split = source_split
        seq.source_sequence = source_sequence
        seq.target_status = target_status
        seq.keyframe_indices = keyframe_indices
        seq.cognitivebench_meta = meta
        return seq

    def _read_meta(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _read_keyframes(self, path):
        indices = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    indices.append(int(line))
        return set(indices)

    def _resolve_frames(self, source_dataset, sequence_name, source_split, expected_len):
        if source_dataset == "lasot":
            class_name = sequence_name.split("-")[0]
            frames_path = Path(self.env_settings.lasot_path) / class_name / sequence_name / "img"
            return [str(frames_path / f"{i:08d}.jpg") for i in range(1, expected_len + 1)]

        if source_dataset == "tnl2k":
            frames_path = Path(self.env_settings.tnl2k_path) / sequence_name / "imgs"
            frame_list = sorted(os.listdir(frames_path))
            return [str(frames_path / frame) for frame in frame_list]

        if source_dataset == "mgit":
            mgit_root = Path(self.env_settings.videocube_path)
            frames_path = mgit_root / "data" / source_split / sequence_name / f"frame_{sequence_name}"
            frame_list = [f for f in os.listdir(frames_path) if f.endswith(".jpg")]
            frame_list.sort(key=lambda f: int(os.path.splitext(f)[0]))
            return [str(frames_path / frame) for frame in frame_list]

        raise ValueError(f"Unsupported CognitiveBench source_dataset: {source_dataset}")

    def _resolve_language(self, source_dataset, sequence_name):
        try:
            if source_dataset == "lasot":
                class_name = sequence_name.split("-")[0]
                nlp_path = Path(self.env_settings.lasot_path) / class_name / sequence_name / "nlp.txt"
                if nlp_path.is_file():
                    return nlp_path.read_text(encoding="utf-8").strip()

            if source_dataset == "tnl2k":
                lang_path = Path(self.env_settings.tnl2k_path) / sequence_name / "language.txt"
                if lang_path.is_file():
                    return lang_path.read_text(encoding="utf-8").strip()

            if source_dataset == "mgit":
                desc_path = Path(self.env_settings.videocube_path) / "attribute" / "description" / f"{sequence_name}.json"
                if desc_path.is_file():
                    with open(desc_path, "r", encoding="utf-8") as f:
                        desc_data = json.load(f)
                    story = desc_data.get("story", {})
                    if "story_1" in story:
                        return story["story_1"].get("description", "").strip()
        except Exception:
            return None

        return None

    def __len__(self):
        return len(self.sequence_list)

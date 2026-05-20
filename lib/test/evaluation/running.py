import numpy as np
import multiprocessing
import os
import sys
import json
from itertools import product
from collections import OrderedDict
from lib.test.evaluation import Sequence, Tracker
import torch


def _save_tracker_output(seq: Sequence, tracker: Tracker, output: dict):
    """Saves the output of the tracker."""

    if not os.path.exists(tracker.results_dir):
        print("create tracking result dir:", tracker.results_dir)
        os.makedirs(tracker.results_dir)
    # if seq.dataset in ['trackingnet', 'got10k', 'lasot', 'lasot_extension_subset', 'otb', 'uav', 'nfs', 'tnl2k']:
    if not os.path.exists(os.path.join(tracker.results_dir, seq.dataset)):
        os.makedirs(os.path.join(tracker.results_dir, seq.dataset))
    '''2021.1.5 create new folder for these three datasets'''
    # if seq.dataset in ['trackingnet', 'got10k', 'lasot', 'lasot_extension_subset', 'otb', 'uav', 'nfs', 'tnl2k']:
    base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
    # else:
    #     base_results_path = os.path.join(tracker.results_dir, seq.name)

    def save_bb(file, data):
        tracked_bb = np.asarray(data, dtype=np.float64)
        if np.isnan(tracked_bb).any():
            np.savetxt(file, tracked_bb, delimiter='	', fmt='%.6f')
        elif np.allclose(tracked_bb, np.round(tracked_bb)):
            np.savetxt(file, tracked_bb.astype(int), delimiter='	', fmt='%d')
        else:
            np.savetxt(file, tracked_bb, delimiter='	', fmt='%.6f')

    def save_time(file, data):
        exec_times = np.array(data).astype(float)
        np.savetxt(file, exec_times, delimiter='\t', fmt='%f')

    def save_score(file, data):
        scores = np.array(data).astype(float)
        np.savetxt(file, scores, delimiter='\t', fmt='%.2f')

    def _convert_dict(input_dict):
        data_dict = {}
        for elem in input_dict:
            for k, v in elem.items():
                if k in data_dict.keys():
                    data_dict[k].append(v)
                else:
                    data_dict[k] = [v, ]
        return data_dict

    def _to_builtin(val):
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, np.generic):
            return val.item()
        if isinstance(val, (list, tuple)):
            return [_to_builtin(v) for v in val]
        if isinstance(val, dict):
            return {k: _to_builtin(v) for k, v in val.items()}
        return val

    def _status_name(value):
        if value is None:
            return None
        try:
            return "present" if int(value) == 1 else "absent"
        except Exception:
            return str(value)

    def _save_cognitivebench_jsonl():
        if seq.dataset != 'cognitivebench':
            return

        jsonl_file = '{}_frames.jsonl'.format(base_results_path)
        target_bbox = output.get('target_bbox', [])
        times = output.get('time', [])
        metadata = output.get('vlm_metadata', [])
        keyframes = getattr(seq, 'keyframe_indices', None)
        target_status = getattr(seq, 'target_status', None)
        target_visible = getattr(seq, 'target_visible', None)
        source_dataset = getattr(seq, 'source_dataset', None)
        source_split = getattr(seq, 'source_split', None)
        source_sequence = getattr(seq, 'source_sequence', seq.name)

        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for frame_id, bbox in enumerate(target_bbox):
                meta = metadata[frame_id] if frame_id < len(metadata) else None
                gt_bbox = None
                if seq.ground_truth_rect is not None and frame_id < len(seq.ground_truth_rect):
                    gt_bbox = _to_builtin(seq.ground_truth_rect[frame_id])

                status_value = None
                if target_status is not None and frame_id < len(target_status):
                    status_value = target_status[frame_id]
                elif target_visible is not None and frame_id < len(target_visible):
                    status_value = 1 if target_visible[frame_id] else 0

                record = {
                    'frame_id': frame_id,
                    'sequence': seq.name,
                    'dataset': seq.dataset,
                    'source_dataset': source_dataset,
                    'source_split': source_split,
                    'source_sequence': source_sequence,
                    'image_path': seq.frames[frame_id] if frame_id < len(seq.frames) else None,
                    'is_keyframe': bool(frame_id in keyframes) if keyframes is not None else None,
                    'gt_bbox': gt_bbox,
                    'gt_target_status': _status_name(status_value),
                    'pred_bbox': _to_builtin(bbox),
                    'time': _to_builtin(times[frame_id]) if frame_id < len(times) else None,
                }

                if meta:
                    record['vlm_output'] = _to_builtin(meta)
                    if isinstance(meta, dict) and meta.get('skipped'):
                        record['skipped'] = True
                        record['skip_reason'] = meta.get('skip_reason')

                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(f"Saved CognitiveBench frame JSONL to {jsonl_file}")

    if seq.dataset == 'cognitivebench':
        _save_cognitivebench_jsonl()
        return

    # ========== 保存 VLM 元数据（新增）==========
    if 'vlm_metadata' in output and output['vlm_metadata']:
        metadata_file = '{}_full.json'.format(base_results_path)

        # 构建完整的跟踪数据结构
        full_data = {
            'sequence': seq.name,
            'dataset': seq.dataset,
            'tracker': tracker.name,
            'parameter': tracker.parameter_name,
            'frames': []
        }

        # 逐帧添加数据
        for frame_id, (bbox, metadata) in enumerate(zip(output['target_bbox'], output['vlm_metadata'])):
            frame_data = {
                'frame_id': frame_id,
                'bbox': bbox if isinstance(bbox, list) else bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox)
            }

            # 添加 VLM 元数据
            if metadata:
                frame_data.update({
                    'target_status': metadata.get('target_status', ''),
                    'environment_status': metadata.get('environment_status', []),
                    'cognition_chain': metadata.get('cognition_chain', ''),
                    'confidence': float(metadata.get('confidence', 0.0))
                })

            full_data['frames'].append(frame_data)

        # 保存为 JSON
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False)

        print(f"Saved VLM metadata to {metadata_file}")
    # ==========================================

    for key, data in output.items():
        # If data is empty
        if not data:
            continue

        if key == 'target_bbox':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        if key == 'all_boxes':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_boxes.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}_all_boxes.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        if key == 'all_scores':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_scores.txt'.format(base_results_path, obj_id)
                    save_score(bbox_file, d)
            else:
                # Single-object mode
                print("saving scores...")
                bbox_file = '{}_all_scores.txt'.format(base_results_path)
                save_score(bbox_file, data)

        elif key == 'time':
            if isinstance(data[0], dict):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    timings_file = '{}_{}_time.txt'.format(base_results_path, obj_id)
                    save_time(timings_file, d)
            else:
                timings_file = '{}_time.txt'.format(base_results_path)
                save_time(timings_file, data)


def run_sequence(seq: Sequence, tracker: Tracker, debug=False, num_gpu=8, run_tag=None):
    """Runs a tracker on a sequence."""
    '''2021.1.2 Add multiple gpu support'''
    if run_tag is not None:
        seq.dataset = run_tag

    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    def _results_exist():
        if seq.object_ids is None:
            # if seq.dataset in ['trackingnet', 'got10k', 'lasot', 'lasot_extension_subset', 'otb', 'uav', 'nfs', 'tnl2k']:
            base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
            if seq.dataset == 'cognitivebench':
                jsonl_file = '{}_frames.jsonl'.format(base_results_path)
                return _cognitivebench_jsonl_complete(jsonl_file, len(seq.frames))
            bbox_file = '{}.txt'.format(base_results_path)
            # else:
            #     bbox_file = '{}/{}.txt'.format(tracker.results_dir, seq.name)
            return os.path.isfile(bbox_file)
        else:
            bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

    def _cognitivebench_jsonl_complete(jsonl_file, expected_frames):
        if not os.path.isfile(jsonl_file):
            return False

        line_count = 0
        last_line = None
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        line_count += 1
                        last_line = line

            if line_count != expected_frames:
                print(f"[CognitiveBench] Incomplete result {jsonl_file}: "
                      f"{line_count}/{expected_frames} frames, rerunning")
                return False

            if last_line is None:
                return False
            last_record = json.loads(last_line)
            if int(last_record.get('frame_id', -1)) != expected_frames - 1:
                print(f"[CognitiveBench] Invalid last frame in {jsonl_file}: "
                      f"{last_record.get('frame_id')} != {expected_frames - 1}, rerunning")
                return False

            return True
        except Exception as e:
            print(f"[CognitiveBench] Invalid result {jsonl_file}: {e}, rerunning")
            return False

    if _results_exist() and not debug:
        print('FPS: {}'.format(-1))
        return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

    if debug:
        output = tracker.run_sequence(seq, debug=debug)
    else:
        try:
            output = tracker.run_sequence(seq, debug=debug)
        except Exception as e:
            print(e)
            return

    sys.stdout.flush()

    if isinstance(output['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output['time']])
        num_frames = len(output['time'])
    else:
        exec_time = sum(output['time'])
        num_frames = len(output['time'])

    print('FPS: {}'.format(num_frames / exec_time))

    _save_tracker_output(seq, tracker, output)


def run_dataset(dataset, trackers, debug=False, threads=0, num_gpus=8, run_tag=None):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
    """
    multiprocessing.set_start_method('spawn', force=True)

    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))

    multiprocessing.set_start_method('spawn', force=True)

    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        for seq in dataset:
            for tracker_info in trackers:
                run_sequence(seq, tracker_info, debug=debug, num_gpu=num_gpus, run_tag=run_tag)
    elif mode == 'parallel':
        param_list = [(seq, tracker_info, debug, num_gpus, run_tag) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print('Done')
